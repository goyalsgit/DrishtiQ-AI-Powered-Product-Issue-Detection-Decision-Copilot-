"""Community Cloud entrypoint. Local development continues to use root app.py."""
import logging
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['DRISHTIQ_PUBLIC_DEMO'] = '1'
os.environ.setdefault('DRISHTIQ_DB_PATH', str(ROOT / 'data/cloud-demo.sqlite'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import streamlit as st
from src.cloud_bootstrap import prepare_cloud_demo


@st.cache_resource(show_spinner='Preparing public reviews and search. First startup may take a few minutes…')
def prepare():
    return prepare_cloud_demo()


try:
    prepare()
except Exception:
    logging.getLogger(__name__).exception('Public demo startup failed')
    st.error('The public demo could not start. The owner can inspect Community Cloud logs and reboot to retry.')
    st.stop()

runpy.run_path(str(ROOT / 'app.py'), run_name='__main__')
