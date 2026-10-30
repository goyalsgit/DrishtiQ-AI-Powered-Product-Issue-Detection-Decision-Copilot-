"""Local backend capacity diagnostic, not a live-user/LLM or relevance benchmark.
Example: python -m scripts.benchmark_capacity --rows 10000 --requests 240
Uses synthetic copies of public review texts in temporary isolated databases.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import resource
import tempfile
import threading
from time import perf_counter

import pandas as pd
import torch
from src.database import ReviewStore
from src.embeddings import ReviewSearch, load_model, MODEL_NAME, INDEX_VERSION

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ['What problems do people report with packaging?',
             'Does the product cause skin irritation?',
             'What do reviewers say about the fragrance?',
             'Does the item match its photograph?',
             'Is the product good value for money?',
             'Are there problems with the spray nozzle?']


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered)-1)*q
    lower = int(index)
    return ordered[lower] + (ordered[min(lower+1, len(ordered)-1)]-ordered[lower])*(index-lower)


def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--requests', type=int, default=240)
    parser.add_argument('--workers', type=int, nargs='+', default=[1,5,10,20])
    args = parser.parse_args()
    if not 100 <= args.rows <= 100000 or args.requests < max(args.workers) or min(args.workers) < 1:
        parser.error('Use 100–100000 rows and requests >= the largest positive worker count.')
    provenance = json.loads((ROOT / 'data/amazon_beauty_provenance.json').read_text())
    originals = ReviewStore().read(provenance['dataset_id'])
    if originals.empty:
        raise SystemExit('Import the public sample first.')
    rng = random.Random(42)
    rows=[]
    for i in range(args.rows):
        source = originals.iloc[rng.randrange(len(originals))]
        # Every tenth review shares one hot scope; the remainder spans 216 scopes.
        product, month = ('HOT-PRODUCT', '2026-01') if i % 10 == 0 else (f'P{i%20:03}', f'2026-{(i//20)%12+1:02}')
        rows.append({'review_id':f'LOAD-{i:08}', 'product':product, 'date':month+'-15',
                     'rating':int(source.rating), 'text':source.text})
    frame=pd.DataFrame(rows)
    del rows
    report={'created_utc':datetime.now(timezone.utc).isoformat(), 'rows':args.rows,
            'method':'Synthetic resampling of public review texts with distinct IDs; actual MiniLM encodings and Chroma indexing.',
            'source_dataset':provenance['dataset_id'], 'seed':42,
            'machine':{'platform':platform.platform(),'logical_cpus':os.cpu_count(),
                       'python':platform.python_version(),'torch_threads':torch.get_num_threads()},
            'packages':{name:version(name) for name in ['torch','sentence-transformers','chromadb']},
            'model':MODEL_NAME,'index_version':INDEX_VERSION,
            'limitations':['No HTTP, browser rendering, Gemini calls, concurrent ingestion, or writes under query load.',
                           'Synthetic repeated text does not measure natural-corpus relevance or establish maximum capacity.',
                           'Closed-loop workers start together, issue repeated queries without think time, and do not simulate arrival queues.',
                           'Query latencies exclude pool scheduling, cold starts and indexing. One trial per concurrency level.',
                           '25% of queries target the hot scope holding 10% of reviews; remaining queries use populated smaller scopes.',
                           'Read-only consistency checks validate count, unique IDs and scope membership, not semantic relevance.'],
            'loads':[]}
    output=ROOT / f'reports/capacity_{args.rows}.json'
    with tempfile.TemporaryDirectory(prefix='drishtiq-load-') as directory:
        work=Path(directory)
        store=ReviewStore(work/'reviews.sqlite')
        t=perf_counter(); dataset, excluded=store.ingest(frame,'SYNTHETIC load test')
        report['ingest_seconds']=round(perf_counter()-t,3)
        assert excluded==0
        del frame
        t=perf_counter(); model=load_model(); report['model_load_seconds']=round(perf_counter()-t,3)
        search=ReviewSearch(model,work/'chroma')
        print(f'Indexing {args.rows:,} synthetic review records using real embeddings…',flush=True)
        t=perf_counter(); chunks=search.build(store,dataset); elapsed=perf_counter()-t
        report.update(index_seconds=round(elapsed,3),passages=chunks,index_reviews_per_second=round(args.rows/elapsed,2))
        catalog=store.catalog(dataset)
        scopes=[(r.product,r.month,int(r.total_reviews)) for r in catalog.itertuples()]
        hot=next(s for s in scopes if s[0]=='HOT-PRODUCT')
        normal=[s for s in scopes if s[0]!='HOT-PRODUCT']
        report['scope_count']=len(scopes);report['hot_scope_reviews']=hot[2]
        report['normal_scope_review_range']=[min(s[2] for s in normal),max(s[2] for s in normal)]
        with store.connect() as con:
            allowed={}
            for review_id,product,month in con.execute('SELECT review_id,product,month FROM reviews WHERE dataset_id=?',(dataset,)):
                allowed.setdefault((product,month),set()).add(review_id)
        for product,month,count in [hot,normal[0]]:
            search.search(store,dataset,product,month,QUESTIONS[0],limit=6)
        for workers in args.workers:
            barrier=threading.Barrier(workers)
            per_worker=[args.requests//workers + int(i<args.requests%workers) for i in range(workers)]
            def worker(worker_id):
                timings=[]; errors=[]; sql_times=[]
                barrier.wait()
                for i in range(per_worker[worker_id]):
                    number=i*workers+worker_id
                    product,month,count=hot if number%4==0 else normal[number%len(normal)]
                    t=perf_counter()
                    try:
                        a=perf_counter();metrics=store.metrics(dataset,product,month);sql_times.append((perf_counter()-a)*1000)
                        hits=search.search(store,dataset,product,month,QUESTIONS[number%len(QUESTIONS)],limit=6)
                        ids=[h['review_id'] for h in hits]
                        if metrics[0]!=count or len(ids)!=6 or len(ids)!=len(set(ids)) or not set(ids)<=allowed[(product,month)]:
                            raise ValueError('Scope/count/duplicate result check failed')
                    except Exception as exc:
                        errors.append({'type':type(exc).__name__,'message':str(exc)[:180]})
                    timings.append((perf_counter()-t)*1000)
                return timings,errors,sql_times
            t=perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes=list(pool.map(worker,range(workers)))
            elapsed=perf_counter()-t
            timings=[v for outcome in outcomes for v in outcome[0]]
            errors=[v for outcome in outcomes for v in outcome[1]]
            sql=[v for outcome in outcomes for v in outcome[2]]
            result={'workers':workers,'requests':len(timings),'completed':len(timings)-len(errors),
                    'error_count':len(errors),'error_examples':errors[:3],
                    'wall_seconds':round(elapsed,3),'successful_requests_per_second':round((len(timings)-len(errors))/elapsed,2),
                    'p50_ms':round(percentile(timings,.5),2),'p95_ms':round(percentile(timings,.95),2),
                    'p99_ms':round(percentile(timings,.99),2),'max_ms':round(max(timings),2),
                    'sql_p95_ms':round(percentile(sql,.95),2)}
            report['loads'].append(result)
            output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(result),flush=True)
        report['sqlite_bytes']=store.path.stat().st_size
        report['chroma_disk_bytes']=folder_bytes(work/'chroma')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['process_peak_rss_mib']=round(rss/(1024**2 if platform.system()=='Darwin' else 1024),1)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Finished: {output}; peak process RSS {report["process_peak_rss_mib"]} MiB',flush=True)


if __name__=='__main__':
    main()
