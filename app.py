import json
import pandas as pd
import streamlit as st
from src.analytics import topics, retrieve, save_decision, history
from src.database import ReviewStore
from src.recommendations import validate_recommendation
from pathlib import Path

st.set_page_config(page_title='DristiQ | Customer Intelligence', page_icon='◈', layout='wide')
st.markdown('''<style>.stApp{background:#0b1120;color:#e5edf9}h1{letter-spacing:-2px}div[data-testid="stMetric"]{background:#152238;padding:20px;border-radius:14px;border:1px solid #263850}</style>''', unsafe_allow_html=True)
st.caption('DRISTIQ / CUSTOMER FEEDBACK INTELLIGENCE')
st.title('Understand what your customers are saying.')
st.write('Ask a question, inspect the evidence, and decide what to investigate next.')
store = ReviewStore()
# Seed the labelled demo only if there are no saved datasets yet.
if store.datasets().empty:
    store.ingest(pd.read_csv(Path(__file__).parent / 'data/sample_reviews.csv'),
                 'Fictional learning dataset', is_demo=True)
with st.sidebar:
    st.header('◈ DristiQ')
    with st.form('import_reviews'):
        upload = st.file_uploader('Upload customer reviews', type='csv')
        st.caption('Columns: review_id, product, date, rating, text. Dates: YYYY-MM-DD. Use a consistent product identifier in product.')
        import_clicked = st.form_submit_button('Validate and save dataset')
    if import_clicked:
        if upload is None:
            st.error('Choose a CSV first.')
        else:
            try:
                dataset_id, removed = store.ingest(
                    pd.read_csv(upload, dtype={'review_id': str, 'product': str}), upload.name)
                st.session_state['dataset_choice'] = dataset_id
                st.success(f'Saved dataset. Excluded {removed} invalid or duplicate rows.')
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))
    datasets = store.datasets().set_index('id')
    dataset_id = st.selectbox('Saved dataset', datasets.index.tolist(), key='dataset_choice',
        format_func=lambda key: f'{datasets.loc[key, "name"]} · {key[:8]}')
    info = datasets.loc[dataset_id]
    st.download_button('Download sample CSV', (Path(__file__).parent / 'data/sample_reviews.csv').read_bytes(), 'sample_reviews.csv', 'text/csv')
df = store.read(dataset_id)
if info['is_demo']:
    st.info('DEMO DATA • Fictional reviews. All insights below describe this synthetic dataset.')
else:
    st.caption(f'Saved source: {info["name"]} · {len(df):,} validated reviews. Uploaded data provenance is supplied by the uploader.')
    provenance_path = Path(__file__).parent / 'data/amazon_beauty_provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get('dataset_id') == dataset_id:
            st.info('PUBLIC HISTORICAL SAMPLE • These reviews are not a random or complete sample. Metrics describe the saved subset only. Product labels are original Amazon IDs.')
            with st.expander('Data source and sampling method'):
                st.write(provenance['sampling'])
                st.markdown(f'[Original dataset]({provenance["source"]})')
                st.caption(f'Source revision: {provenance["revision"]}')
with st.sidebar:
    product = st.selectbox('Product', sorted(df['product'].unique()))
    selected = df[df['product'].eq(product)]
    months = sorted(selected['month'].unique())
    current = st.selectbox('Analysis month', months, index=len(months)-1)
    earlier = [m for m in months if m < current]
    baseline = st.selectbox('Compare with', earlier, index=len(earlier)-1) if earlier else None
    minimum = st.number_input('Minimum reviews per period', 5, 500, 20)
    threshold = st.slider('Alert threshold (percentage points)', 1, 50, 10)
    st.caption('Low rating = 1 or 2 stars. Topic tags use keywords, not AI sentiment analysis.')
now = selected[selected['month'].eq(current)]
before = selected[selected['month'].eq(baseline)]
n, avg, neg, share = store.metrics(dataset_id, product, current)
pn, _, _, previous = store.metrics(dataset_id, product, baseline)
delta = share - previous if previous is not None else None
cols = st.columns(4)
cols[0].metric('Reviews analysed', n)
cols[1].metric('Average rating', f'{avg:.2f} / 5')
cols[2].metric('Low-rating share', f'{share:.1f}%', f'{delta:+.1f} pp' if delta is not None else None, delta_color='inverse')
cols[3].metric('Low-rating reviews', neg)
investigation, overview, evidence, approval, learn = st.tabs(
    ['AI Investigation', 'Overview', 'Review evidence', 'Decisions', 'How it works'])
with investigation:
    from src.dashboard_ai import render_investigation
    render_investigation(store, dataset_id, product, current, baseline, int(minimum))
with overview:
    st.subheader(f'{product} · {current}')
    if delta is None:
        st.info('Choose a later month to compare with an earlier period.')
    elif min(n, pn) < minimum:
        st.warning(f'Insufficient sample: {pn} baseline and {n} current reviews. Need {minimum} in each period.')
    elif delta >= threshold:
        st.warning(f'Low-rating alert: +{delta:.1f} percentage points ({previous:.1f}% → {share:.1f}%). This is a rule-based alert, not a statistical significance test.')
    else:
        st.success('The increase does not meet the selected alert threshold.')
    left, right = st.columns(2)
    with left:
        st.write('Monthly low-rating share (%)')
        st.line_chart(selected.groupby('month')['negative'].mean().mul(100))
    tagged = topics(now)
    with right:
        st.write('Keyword-tagged complaints in low-rating reviews')
        st.bar_chart(pd.Series({k: len(v) for k, v in tagged.items()}, name='Mentions'))
    st.caption('One review can match several topics. Keyword matches can miss nuance or negation. Review the evidence before acting.')
with evidence:
    st.subheader('Find the words behind the numbers')
    mode = st.radio('Search method', ['Keyword', 'Semantic'], horizontal=True)
    query = st.text_input('Search reviews', placeholder='Try: liquid spilled during shipping')
    if mode == 'Keyword':
        matches = retrieve(now, query) if query.strip() else now
        st.dataframe(matches[['review_id', 'date', 'rating', 'text']], width='stretch', hide_index=True)
    else:
        st.caption('Local semantic search within this dataset, product, and month. Similarity is not confidence; results may be irrelevant. English baseline model.')
        try:
            from src.embeddings import ReviewSearch, load_model

            @st.cache_resource
            def semantic_search():
                return ReviewSearch(load_model())

            search = semantic_search()
            if not search.ready(dataset_id):
                st.info('Prepare this dataset for semantic search once. Larger datasets take longer.')
                if st.button('Build semantic index'):
                    with st.spinner('Embedding review passages locally…'):
                        passages = search.build(store, dataset_id)
                    st.success(f'Saved {passages:,} searchable passages.')
            if search.ready(dataset_id):
                st.caption('Index ready. Full source reviews remain in SQLite.')
                if query.strip():
                    with st.spinner('Finding relevant reviews…'):
                        hits = search.search(store, dataset_id, product, current, query)
                    if hits:
                        st.dataframe(pd.DataFrame(hits), width='stretch', hide_index=True)
                    else:
                        st.info('No reviews found within the selected scope.')
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            st.error(f'Semantic search unavailable: {exc}')
            st.caption('Setup: install requirements-chroma.txt and run lesson 04 once to download the model.')
    st.download_button('Export selected reviews', now.to_csv(index=False), 'dristiq_reviews.csv', 'text/csv')
with approval:
    st.subheader('Review a proposed action')
    topic = st.selectbox('Complaint to investigate', list(tagged))
    supporting = tagged[topic]
    if supporting.empty:
        st.info('No low-rating reviews matched this topic in this period.')
    else:
        action = f'Request a {topic.lower()} review for {product} ({current})'
        proposal = {
            'product': product,
            'issue': f'Low-rating reviews contain keywords associated with {topic.lower()}.',
            'evidence_ids': supporting['review_id'].tolist(),
            'recommended_action': action,
        }
        try:
            summary = validate_recommendation(proposal, supporting['review_id'].tolist())
        except ValueError as exc:
            st.error(f'Recommendation validation failed: {exc}')
            st.stop()
        st.write(action)
        with st.expander('Inspect validated recommendation'):
            st.json(summary.model_dump())
        st.caption('Rule-based suggestion. Saving a decision records it locally; it does not send messages or execute an external action.')
        st.dataframe(supporting[['review_id', 'rating', 'text']], hide_index=True, width='stretch')
        reviewer = st.text_input('Reviewer name')
        decision = st.radio('Decision', ['Approve', 'Reject'], horizontal=True)
        context = {'dataset': dataset_id, 'product': product, 'month': current, 'baseline': baseline,
                   'topic': topic, 'evidence': supporting['review_id'].tolist()}
        if st.button('Save decision', type='primary', disabled=not reviewer.strip()):
            validate_recommendation(proposal, supporting['review_id'].tolist())
            save_decision(context, action, decision, reviewer.strip())
            st.success('Decision recorded. An existing decision for this exact proposal is preserved.')
    st.subheader('Persistent decision history')
    st.dataframe(history(dataset_id), width='stretch', hide_index=True)
with learn:
    st.markdown('''**1. Pandas cleans the CSV.** Each row is one review. We reject invalid ratings and missing fields.

**2. SQLite stores immutable dataset snapshots.** Reimporting the same validated data does not duplicate it. SQL calculates metrics directly from the saved reviews.

**3. Period comparison detects a change.** We require enough reviews before raising a threshold alert. A change alone does not prove its cause.

**4. Semantic search retrieves evidence.** MiniLM embeds review passages; Chroma saves and searches them. Dataset, product, and month filters keep results in scope. Complaint charts still use keyword tags, not AI classification.

**5. SQLite saves human decisions.** The decision includes dataset identity and supporting review IDs, so it survives an app restart.

**6. AI Investigation connects Gemini to the tools.** Enter a key and ask a question.
LangGraph runs the model/tool loop, SQL supplies metrics, and MiniLM + Chroma find
review evidence. The report, evidence, and tool activity appear together.

**What is still pending:** structured AI report validation, stronger evidence
relevance checks, and persistent approval of AI investigations. The Decisions tab
currently records rule-based proposals; it does not approve an AI report.''')
