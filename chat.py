import sys
from datetime import date
import numpy as np

from config import ollama_client, CHAT_MODEL, DB_FILE, EMBED_MODEL, MIN_SCORE, QUERY_PREFIX, SHOW_DEBUG, TOP_K
from db import connect, from_blob, get_meta

SYSTEM_PROMPT = """Kamu adalah asisten virtual yang ramah, sopan, dan sangat membantu dari Kopi Nusantara. 
Jawablah pertanyaan pelanggan selayaknya barista atau staf kafe yang sedang berbicara langsung dengan mereka.

Aturan Wajib:
1. BERBICARA SECARA NATURAL: Langsung berikan jawaban kepada pelanggan. DILARANG KERAS menyebutkan kata "Konteks", "Dokumen", "Database", "Sistem", atau "Berdasarkan data". 
   (SALAH: "Menurut konteks, hari Minggu buka jam 07:00." -> BENAR: "Halo! Untuk hari Minggu, kami buka mulai pukul 07:00 pagi ya.")
2. Sopan dan Hangat: Gunakan bahasa Indonesia yang natural dan ramah. Kamu boleh menggunakan sapaan seperti "Halo" atau "Kak".
3. Penalaran Logis: Lakukan penalaran sederhana dari informasi yang kamu miliki (contoh: mengubah "setengah 11 malam" menjadi 22:30, atau mengkategorikan menu termurah).
4. Sesuai Fakta: Dilarang mengarang informasi (halusinasi) atau memberikan harga/menu/jadwal yang tidak ada di informasi rahasiamu.
5. Menangani Pertanyaan di Luar Topik: Jika informasi yang ditanyakan BENAR-BENAR TIDAK ADA, jangan bilang "tidak ada di data saya". Tetap berakting sebagai staf dan jawab: "Mohon maaf Kak, untuk informasi tersebut belum tersedia saat ini." atau "Wah, maaf Kak, saya belum mendapat info terkait hal itu."
6. Jika ada dua informasi yang bertentangan, selalu gunakan informasi dengan pembaruan yang paling baru."""

def load_index():
    if not DB_FILE.exists(): sys.exit('rag.db belum ada. Jalankan: python ingest.py')
    conn = connect()
    if get_meta(conn, 'embed_model') != EMBED_MODEL: sys.exit('Model embedding berubah. Jalankan ulang ingest.py')
    rows = conn.execute('SELECT source, judul, diperbarui, text, embedding FROM chunks').fetchall()
    conn.close()
    
    records = [dict(r) for r in rows]
    matrix = np.vstack([from_blob(r['embedding']) for r in records])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return records, matrix

def retrieve(question, records, matrix):
    response = ollama_client.embed(model=EMBED_MODEL, input=QUERY_PREFIX + question)
    query = np.array(response['embeddings'][0], dtype=np.float32)
    query = query / np.linalg.norm(query)
    scores = matrix @ query
    best = np.argsort(scores)[::-1][:TOP_K]
    return [(records[i], float(scores[i])) for i in best]

def build_prompt(question, hits):
    blocks = [r['text'] + f"\n(Diperbarui: {r['diperbarui']})" for r, s in hits]
    return f"KONTEKS:\n{chr(10).join(blocks)}\n\nPERTANYAAN: {question}"

def main():
    records, matrix = load_index()
    system = SYSTEM_PROMPT + '\nTanggal hari ini: ' + date.today().isoformat()
    print(f'=== Chatbot CLI RAG (Llama3.2 & BGE-M3) | ketik exit untuk keluar ===')

    while True:
        try:
            question = input('\nKamu > ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question: continue
        if question.lower() in ('exit', 'quit', 'keluar'): break

        top = retrieve(question, records, matrix)
        if SHOW_DEBUG:
            for r, s in top: print(f"  [debug] {r['judul']} | skor={s:.2f} | diperbarui={r['diperbarui']}")

        hits = [h for h in top if h[1] >= MIN_SCORE]
        if not hits:
            print('Bot  > Maaf, saya tidak menemukan informasi itu di data saya.')
            continue

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': build_prompt(question, hits)},
        ]

        print('Bot  > ', end='', flush=True)
        for chunk in ollama_client.chat(model=CHAT_MODEL, messages=messages, stream=True):
            print(chunk['message']['content'], end='', flush=True)
        # print('\n[Sumber: ' + ', '.join(sorted({r['source'] for r, s in hits})) + ']')

if __name__ == '__main__':
    main()