"""How DrishtiQ works, in 7 tiny steps. Each step is the simple version of a real file.

Run:  .venv-agent/bin/python -m lessons.12_how_drishtiq_works
      .venv-agent/bin/python -m lessons.12_how_drishtiq_works --gemini   (also step 7, uses your key)
Uses 8 made-up reviews and temporary files. Your real data is not touched.
"""
import sqlite3
import sys
import tempfile

REVIEWS = [  # review_id, product, rating, text
    ('r1', 'Tablet', 1, 'The battery dies in two hours.'),
    ('r2', 'Tablet', 2, 'Battery does not hold a charge anymore.'),
    ('r3', 'Tablet', 1, 'Screen cracked when it arrived.'),
    ('r4', 'Tablet', 5, 'Great price and my kids love it.'),
    ('r5', 'Echo', 2, 'Alexa does not understand my questions.'),
    ('r6', 'Echo', 1, 'It cannot answer simple questions.'),
    ('r7', 'Echo', 5, 'Plays music very well.'),
    ('r8', 'Tablet', 2, 'The charging port broke after a month.'),
]

print('\n=== STEP 1: Save reviews in a database (real file: src/database.py) ===')
db = sqlite3.connect(':memory:')
db.execute('CREATE TABLE reviews (review_id TEXT, product TEXT, rating INT, text TEXT)')
db.executemany('INSERT INTO reviews VALUES (?,?,?,?)', REVIEWS)
print('Saved', db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0], 'reviews')

print('\n=== STEP 2: SQL counts exact numbers. The AI never does this. ===')
rows = db.execute('''SELECT product, COUNT(*), SUM(rating <= 2) FROM reviews
                     GROUP BY product ORDER BY SUM(rating <= 2) DESC''').fetchall()
for product, total, low in rows:
    print(f'{product}: {total} reviews, {low} are 1-2 stars')
# The "?" is filled in safely by SQLite. User text can never change the SQL itself.
print('Tablet low ratings:', db.execute(
    'SELECT COUNT(*) FROM reviews WHERE product=? AND rating<=2', ('Tablet',)).fetchone()[0])

print('\n=== STEP 3: Embeddings - text becomes 384 numbers (real file: src/embeddings.py) ===')
from src.embeddings import load_model
model = load_model()                                   # all-MiniLM-L6-v2, runs on your laptop
texts = [r[3] for r in REVIEWS]
vectors = model.encode(texts, normalize_embeddings=True)
print('One review is now', len(vectors[0]), 'numbers. First 5:', [round(float(x), 3) for x in vectors[0][:5]])
# Similar meaning -> similar numbers. The dot product of normalised vectors = cosine similarity.
import numpy as np
print('"battery dies" vs "does not hold a charge":', round(float(vectors[0] @ vectors[1]), 2))
print('"battery dies" vs "plays music"           :', round(float(vectors[0] @ vectors[6]), 2))

print('\n=== STEP 4: Search by meaning with Chroma (real file: src/embeddings.py) ===')
import chromadb
chroma = chromadb.PersistentClient(path=tempfile.mkdtemp())
col = chroma.create_collection('demo', metadata={'hnsw:space': 'cosine'})
col.add(ids=[r[0] for r in REVIEWS], documents=texts, embeddings=vectors.tolist(),
        metadatas=[{'product': r[1]} for r in REVIEWS])
question = 'Is the power life bad?'                    # note: the word "battery" is not used
hits = col.query(query_embeddings=model.encode([question], normalize_embeddings=True).tolist(),
                 n_results=2, where={'product': 'Tablet'})
print(f'Question: "{question}"')
for text, distance in zip(hits['documents'][0], hits['distances'][0]):
    print(f'  similarity {1 - distance:.2f}: {text}')

print('\n=== STEP 5: Themes - group ALL low-rated reviews (real file: src/themes.py) ===')
from sklearn.cluster import KMeans
low = [i for i, r in enumerate(REVIEWS) if r[2] <= 2]  # only 1-2 star reviews
groups = KMeans(n_clusters=3, n_init=4, random_state=0).fit(vectors[low]).labels_
for g in sorted(set(groups)):
    members = [REVIEWS[low[i]] for i, x in enumerate(groups) if x == g]
    print(f'Theme {g}: {len(members)} reviews ->', [m[3] for m in members])
print('Nothing above was hard-coded: the groups come from the review meanings.')

print('\n=== STEP 6: Check the AI did not invent a quote (real file: src/reporting.py) ===')
def quote_is_real(quote, review_text):
    return quote in review_text
print('Real quote   :', quote_is_real('battery dies in two hours', REVIEWS[0][3]))
print('Invented one :', quote_is_real('battery exploded', REVIEWS[0][3]), '-> removed from the answer')

print('\n=== STEP 7: Tool calling with LangChain + LangGraph (real files: src/tools.py, src/agent.py) ===')
from langchain.tools import tool

@tool
def count_low_ratings(product: str) -> int:
    """Count 1-2 star reviews for a product."""          # Gemini reads this description
    return db.execute('SELECT COUNT(*) FROM reviews WHERE product=? AND rating<=2',
                      (product,)).fetchone()[0]

print('A tool is just a Python function with a description. Called by hand:',
      count_low_ratings.invoke({'product': 'Echo'}))

if '--gemini' not in sys.argv:
    print('\n(Add --gemini to watch Gemini choose this tool itself.)')
    raise SystemExit

import tomllib, pathlib
from langchain.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from src.llm import make_llm

key = tomllib.loads(pathlib.Path('.streamlit/secrets.toml').read_text())['GOOGLE_API_KEY']
gemini = make_llm(api_key=key).bind_tools([count_low_ratings])   # tell Gemini the tool exists

def ask_gemini(state):                     # node 1: Gemini answers OR asks for a tool
    return {'messages': [gemini.invoke(state['messages'])]}

def run_tools(state):                      # node 2: OUR code runs the tool Gemini asked for
    out = []
    for call in state['messages'][-1].tool_calls:
        print(f'  Gemini asked: {call["name"]}({call["args"]})')
        out.append(ToolMessage(str(count_low_ratings.invoke(call['args'])), tool_call_id=call['id']))
    return {'messages': out}

def next_step(state):                      # the arrow: tool requested? run it. Otherwise stop.
    return 'tools' if state['messages'][-1].tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node('gemini', ask_gemini)
graph.add_node('tools', run_tools)
graph.add_edge(START, 'gemini')
graph.add_conditional_edges('gemini', next_step)
graph.add_edge('tools', 'gemini')          # after a tool, go back to Gemini
app = graph.compile()

final = app.invoke({'messages': [HumanMessage('How many low-rated reviews does Echo have?')]})
from src.investigation import answer_text
print('  Gemini answered:', answer_text(final['messages'][-1].content))
