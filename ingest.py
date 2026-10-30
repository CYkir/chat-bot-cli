import csv
import hashlib

from config import ollama_client, BATCH_SIZE, DATA_DIR, DOC_PREFIX, EMBED_MODEL, MAX_CHARS
from db import connect, get_meta, set_meta, to_blob

def parse_front_matter(text):
    meta = {}
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    meta[key.strip()] = value.strip()
            text = parts[2].strip()
    return meta, text

def split_long(text, max_chars=MAX_CHARS):
    parts = []
    current = ''
    for paragraph in text.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph: continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = current + '\n\n' + paragraph if current else paragraph
    if current: parts.append(current)
    return parts

def chunk_markdown(path):
    meta, text = parse_front_matter(path.read_text(encoding='utf-8'))
    doc_title = path.stem
    sections = []
    title = ''
    lines = []
    for line in text.splitlines():
        if line.startswith('# '): doc_title = line[2:].strip()
        elif line.startswith('## '):
            if ''.join(lines).strip():
                sections.append((title, '\n'.join(lines).strip()))
            title = line[3:].strip()
            lines = []
        else: lines.append(line)
    if ''.join(lines).strip():
        sections.append((title, '\n'.join(lines).strip()))
    
    chunks = []
    for title, body in sections:
        label = doc_title + ' > ' + title if title else doc_title
        for part in split_long(body):
            chunks.append({
                'source': path.as_posix(), 'judul': label,
                'kategori': meta.get('kategori', 'tetap'),
                'diperbarui': meta.get('diperbarui', '-'),
                'text': label + '\n' + part,
            })
    return chunks

def chunk_csv(path):
    chunks = []
    label = path.stem
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            facts = '; '.join(f'{k}: {v}' for k, v in row.items() if v)
            chunks.append({
                'source': path.as_posix(), 'judul': label,
                'kategori': 'berubah', 'diperbarui': row.get('diperbarui', '-'),
                'text': label + '\n' + facts,
            })
    return chunks

def make_id(chunk):
    raw = chunk['source'] + '\n' + chunk['text']
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def embed_texts(texts):
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [DOC_PREFIX + t for t in texts[i:i + BATCH_SIZE]]
        response = ollama_client.embed(model=EMBED_MODEL, input=batch, truncate=False)
        vectors += response['embeddings']
        print(f'  embedding {min(i + BATCH_SIZE, len(texts))}/{len(texts)}')
    return vectors

def main():
    files = sorted(list(DATA_DIR.rglob('*.md')) + list(DATA_DIR.rglob('*.csv')))
    if not files:
        print('Tidak ada file .md atau .csv di folder data/')
        return
    
    chunks = []
    for file in files:
        found = chunk_markdown(file) if file.suffix == '.md' else chunk_csv(file)
        chunks += found
    
    for chunk in chunks: chunk['id'] = make_id(chunk)
    chunks = list({c['id']: c for c in chunks}.values())
    current_ids = {c['id'] for c in chunks}
    
    conn = connect()
    if get_meta(conn, 'embed_model') != EMBED_MODEL:
        conn.execute('DELETE FROM chunks')
        set_meta(conn, 'embed_model', EMBED_MODEL)
    
    existing_ids = {row['id'] for row in conn.execute('SELECT id FROM chunks')}
    removed = existing_ids - current_ids
    conn.executemany('DELETE FROM chunks WHERE id = ?', [(i,) for i in removed])
    
    new_chunks = [c for c in chunks if c['id'] not in existing_ids]
    if new_chunks:
        vectors = embed_texts([c['text'] for c in new_chunks])
        rows = [(c['id'], c['source'], c['judul'], c['kategori'], c['diperbarui'], c['text'], to_blob(v)) for c, v in zip(new_chunks, vectors)]
        conn.executemany('INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
    
    conn.commit()
    conn.close()
    print(f'Selesai! {len(chunks)} potongan di database | +{len(new_chunks)} baru | -{len(removed)} dihapus')

if __name__ == '__main__':
    main()