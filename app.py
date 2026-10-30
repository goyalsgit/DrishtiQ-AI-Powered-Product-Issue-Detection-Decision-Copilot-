"""DrishtiQ dashboard.

Tabs:  Overview | Ask reviews | Reviews | Data | Project & quality
Every number is read from SQLite on each page load (no caching), so reviews added
by CSV upload or database sync appear as soon as the page refreshes.
"""
from datetime import date
import os

import pandas as pd
import streamlit as st

from src.dashboard_ai import get_search, render_ai_settings, render_investigation
from src.dashboard_project import render_project_status
from src.database import ReviewStore

PUBLIC_DEMO = os.getenv('DRISHTIQ_PUBLIC_DEMO') == '1'
CSV_HELP = 'Columns: review_id, product, date (YYYY-MM-DD), rating (1-5), text.'

st.set_page_config(page_title='DrishtiQ', page_icon='🔎', layout='wide')
store = ReviewStore()


def read_upload(upload):
    return pd.read_csv(upload, dtype={'review_id': str, 'product': str})


# ---------- First run: empty database ----------
datasets = store.datasets()
if datasets.empty:
    st.title('DrishtiQ')
    st.info('The review database is empty. Load the public sample or import your own CSV. '
            'No demo data is added automatically.')
    if st.button('Load public sample: 932 real Amazon Beauty reviews', type='primary'):
        from src.cloud_bootstrap import ensure_public_sample
        with st.spinner('Downloading and saving the public sample…'):
            ensure_public_sample(store)
        st.rerun()
    with st.form('initial_import'):
        upload = st.file_uploader('Import reviews (CSV)', type='csv', help=CSV_HELP)
        if st.form_submit_button('Save to database') and upload is not None:
            try:
                store.ingest(read_upload(upload), upload.name)
                st.rerun()
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))
    st.stop()


# ---------- Sidebar: dataset and investigation scope ----------
with st.sidebar:
    st.header('DrishtiQ')
    names = datasets.set_index('id')['name']
    dataset_id = st.selectbox('Dataset', datasets['id'].tolist(), key='dataset_choice',
                              format_func=lambda key: names[key])
    catalog = store.catalog(dataset_id)
    if catalog.empty:
        st.warning('This dataset has no reviews yet. Add some in the Data tab.')
        product = current = baseline = None
    else:
        st.subheader('Investigation scope')
        # None = "All products": questions can compare and cover every product.
        product = st.selectbox('Product', [None] + sorted(catalog['product'].unique()), index=1,
                               format_func=lambda p: '🌐 All products' if p is None else p)
        product_rows = (catalog if product is None else catalog[catalog['product'].eq(product)])
        product_rows = product_rows.groupby('month', as_index=False)['total_reviews'].sum()
        months = product_rows['month'].tolist()
        counts = dict(zip(months, product_rows['total_reviews']))
        total = int(product_rows['total_reviews'].sum())
        # "All months" (None) searches the product's whole history.
        current = st.selectbox(
            'Analysis month', [None] + months, index=len(months),
            format_func=lambda m: f'All months ({total} reviews)' if m is None
            else f'{m} ({counts[m]} reviews)')
        # Month comparison is per product; not offered for All products.
        earlier = [m for m in months if current is not None and m < current and product is not None]
        compare = st.checkbox('Compare months', key='ai_compare', disabled=not earlier)
        baseline = (st.selectbox('Compare with', earlier, index=len(earlier) - 1)
                    if compare and earlier else None)
    st.divider()
    render_ai_settings()
    with st.expander('Analysis settings'):
        minimum = int(st.number_input('Minimum reviews per period', 1, 500, 20))
        st.caption('A sample-size rule, not a statistical significance test.')


# ---------- Header ----------
overall = store.summary(dataset_id)
left, right = st.columns([4, 1])
left.title('DrishtiQ')
left.caption('Customer review intelligence: exact SQL statistics plus review evidence. '
             f'Dataset: {names[dataset_id]}')
if right.button('🔄 Refresh data', width='stretch'):
    st.rerun()
right.caption(f"Latest review: {overall['last_date'] or '—'}")


def filter_bar(prefix):
    """Shared filters. Returns keyword arguments for the ReviewStore analytics methods."""
    products = sorted(catalog['product'].unique()) if not catalog.empty else []
    first = date.fromisoformat(overall['first_date'][:10]) if overall['first_date'] else date.today()
    last = date.fromisoformat(overall['last_date'][:10]) if overall['last_date'] else date.today()
    c1, c2, c3 = st.columns([2, 2, 1.4])
    chosen = c1.multiselect('Products', products, key=f'{prefix}_products',
                            placeholder='All products')
    period = c2.date_input('Date range', (first, last), min_value=first, max_value=last,
                           key=f'{prefix}_dates')
    ratings = c3.multiselect('Stars', [1, 2, 3, 4, 5], key=f'{prefix}_stars', placeholder='All')
    start, end = (period if isinstance(period, tuple) and len(period) == 2 else (first, last))
    return {'products': chosen or None, 'start': start.isoformat(),
            'end': end.isoformat() + 'T23:59:59', 'ratings': ratings or None}


def kpi_row(stats):
    cols = st.columns(4)
    cols[0].metric('Reviews', f"{stats['reviews']:,}")
    cols[1].metric('Products', stats['products'])
    cols[2].metric('Average rating',
                   f"{stats['average_rating']:.2f} ★" if stats['average_rating'] else '—')
    share = stats['low_rating_share_percent']
    cols[3].metric('Low-rating share', f'{share:.1f}%' if share is not None else '—',
                   help='Share of 1-2 star reviews. Not a count of a specific complaint.')


def render_company_sync(search):
    """Sync from the company database: Neon Postgres if COMPANY_DB_URL is set, else a local file."""
    from src import company_db
    with st.container(border=True):
        st.markdown('**Company database sync**')
        st.caption(f'Source: {company_db.description()} (table customer_reviews). Only new rows '
                   'are copied into "Company reviews (live)" and only those are indexed.')
        try:
            ready = company_db.exists()
        except Exception as exc:
            st.error(f'Cannot reach the company database ({type(exc).__name__}). Check COMPANY_DB_URL.')
            return
        if not ready:
            if PUBLIC_DEMO:
                st.info('No customer_reviews table yet. Run scripts.load_company_db on your laptop.')
            elif st.button('Create demo company database (1,500 real reviews)'):
                with st.spinner('Creating…'):
                    st.success(f'{company_db.create():,} reviews written. Now click Sync now.')
            return
        if st.button('🔄 Sync now', type='primary'):
            with st.spinner('Copying new rows and indexing them…'):
                _, result = company_db.sync(store, search)
            st.success(f"Fetched {result['fetched']:,} new rows, added {result['added']:,} reviews, "
                       f"indexed {result['passages_indexed']:,} passages. Choose "
                       f"\"{company_db.live_name()}\" in the sidebar (refresh if it is new).")
        with st.form('add_company_review'):
            st.markdown('Simulate a customer posting a review in the company system:')
            sku = st.selectbox('Company product', company_db.products())
            stars = st.slider('Stars', 1, 5, 2)
            body = st.text_area('Review text', max_chars=1000)
            if st.form_submit_button('Post review to company database'):
                if body.strip():
                    row = company_db.add_review(sku, stars, body.strip())
                    st.success(f'Posted as row {row}. It is NOT in DrishtiQ until you click Sync now.')
                else:
                    st.warning('Write some review text first.')

overview, ask, reviews_tab, data_tab, project = st.tabs(
    ['Overview', 'Ask reviews', 'Reviews', 'Data', 'Project & quality'])


# ---------- Overview: all products at a glance ----------
with overview:
    if catalog.empty:
        st.info('No reviews yet.')
    else:
        filters = filter_bar('ov')
        stats = store.summary(dataset_id, **filters)
        kpi_row(stats)
        if stats['reviews'] == 0:
            st.info('No reviews match these filters.')
        else:
            trend = store.monthly_trend(dataset_id, **filters).set_index('month')
            c1, c2 = st.columns([2, 1])
            with c1.container(border=True):
                st.markdown('**Low-rating share by month (%)**')
                st.line_chart(trend['low_rating_share_percent'], height=240)
                st.caption('Months with few reviews swing a lot; check review counts below.')
            with c2.container(border=True):
                st.markdown('**Star rating distribution**')
                dist = store.rating_distribution(dataset_id, **filters)
                st.bar_chart(dist.set_index('rating')['reviews'], height=240)
            with st.container(border=True):
                st.markdown('**Reviews per month**')
                st.bar_chart(trend['reviews'], height=180)
            st.subheader('Products ranked by low-rating share')
            table = store.product_table(dataset_id, **filters)
            table['note'] = table['reviews'].map(
                lambda n: f'small sample (<{minimum})' if n < minimum else '')
            st.dataframe(table, hide_index=True, width='stretch', column_config={
                'product': 'Product',
                'reviews': st.column_config.NumberColumn('Reviews'),
                'average_rating': st.column_config.NumberColumn('Avg ★', format='%.2f'),
                'low_rating_reviews': st.column_config.NumberColumn('1-2 ★ reviews'),
                'low_rating_share_percent': st.column_config.ProgressColumn(
                    'Low-rating share', format='%.1f%%', min_value=0, max_value=100),
                'latest_review': 'Latest review', 'note': 'Note'})
            st.caption('Ask DrishtiQ about a product: choose it in the sidebar, then open Ask reviews.')


# ---------- Ask reviews: AI investigation for one product ----------
with ask:
    if catalog.empty:
        st.info('Add reviews first.')
    else:
        n, avg, low, share = store.metrics(dataset_id, product, current)
        period = current or 'all months'
        st.caption(f'Scope: {product or "all products"} · {period}'
                   + (f' vs {baseline}' if baseline else ''))
        cols = st.columns(3)
        cols[0].metric('Reviews in scope', n)
        cols[1].metric('Average rating', f'{avg:.2f} ★' if avg is not None else '—')
        cols[2].metric('Low-rating share', f'{share:.1f}%' if share is not None else '—')
        if n < minimum:
            st.warning(f'Only {n} reviews in {period}. Answers will be thin. '
                       'Choose "All months" in the sidebar to search the whole product history.')
        render_investigation(store, dataset_id, product, current, baseline, minimum)


# ---------- Reviews: filter, read and export original reviews ----------
with reviews_tab:
    if catalog.empty:
        st.info('No reviews yet.')
    else:
        filters = filter_bar('rv')
        filters['text'] = st.text_input('Contains text', key='rv_text',
                                        placeholder='e.g. leak, smell, broken')
        total = store.summary(dataset_id, **filters)['reviews']
        pages = max(1, (total + 24) // 25)
        c1, c2 = st.columns([1, 3])
        page = c1.number_input('Page', 1, pages, 1, key='rv_page')
        c2.caption(f'{total:,} matching reviews · page {page} of {pages} · 25 per page')
        st.dataframe(store.filtered_page(dataset_id, offset=(int(page) - 1) * 25, **filters),
                     hide_index=True, width='stretch', column_config={
                         'date': 'Date', 'product': 'Product',
                         'rating': st.column_config.NumberColumn('★'),
                         'text': st.column_config.TextColumn('Review', width='large'),
                         'review_id': 'Review ID'})
        st.download_button('Download matching reviews (CSV, max 50,000)',
                           store.filtered_export(dataset_id, **filters).to_csv(index=False),
                           file_name='drishtiq_reviews.csv', mime='text/csv')


# ---------- Data: storage, imports and search index ----------
with data_tab:
    st.write('The application reads the saved SQLite database. You do not need to upload a CSV each time.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Saved reviews', f"{overall['reviews']:,}")
    c2.metric('Products', overall['products'])
    c3.metric('Dataset type', 'Live (grows)' if dataset_id.startswith('live-') else 'Snapshot (fixed)')
    st.caption(f'SQLite file: {store.path}')

    with st.container(border=True):
        st.markdown('**Review search index**')
        search = get_search()
        ready = search.ready(dataset_id)
        st.write('✅ Ready' if ready else '⚠️ Not prepared for this dataset')
        st.caption('MiniLM turns review passages into vectors; Chroma stores them. Runs locally, no Gemini call.')
        if not PUBLIC_DEMO and st.button('Prepare review search', key='ai_prepare', disabled=ready):
            with st.spinner('Indexing reviews…'):
                st.success(f'{search.build(store, dataset_id):,} passages ready.')

    if not PUBLIC_DEMO:
        with st.container(border=True):
            st.markdown('**Add reviews from a CSV file**')
            st.caption(CSV_HELP)
            with st.form('additional_import'):
                upload = st.file_uploader('CSV file', type='csv')
                mode = st.radio('Save as', [
                    'Add to a live dataset (new reviews are appended and indexed)',
                    'New snapshot (a fixed copy that never changes)'])
                live_name = st.text_input('Live dataset name', value='Company reviews (live)')
                submitted = st.form_submit_button('Save to database', type='primary')
            if submitted:
                if upload is None:
                    st.warning('Choose a file first.')
                else:
                    try:
                        frame = read_upload(upload)
                        if mode.startswith('Add'):
                            target = store.create_live_dataset(live_name.strip() or 'Company reviews (live)')
                            added, removed = store.append(target, frame)
                            indexed = search.add_reviews(store, target, added) if added else 0
                            st.success(f'Added {len(added):,} new reviews ({indexed:,} passages indexed). '
                                       f'Skipped {removed} invalid rows; already-saved reviews are ignored.')
                        else:
                            _, removed = store.ingest(frame, upload.name)
                            st.success(f'Snapshot saved. Excluded {removed} invalid or duplicate rows.')
                        st.caption('Select the dataset in the sidebar (refresh if it is new).')
                    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                        st.error(str(exc))

    # On the public host, show sync only when a real Postgres (COMPANY_DB_URL) is configured.
    from src.company_db import database_url
    if not PUBLIC_DEMO or database_url():
        render_company_sync(search)


with project:
    render_project_status()
