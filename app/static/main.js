/* Frontend logic for uploading and rendering results. */

(function () {
  const form = document.getElementById('uploadForm');
  const fileInput = document.getElementById('fileInput');
  const submitBtn = document.getElementById('submitBtn');
  const statusEl = document.getElementById('status');
  const resultsEl = document.getElementById('results');
  const summaryEl = document.getElementById('summary');
  const notesEl = document.getElementById('notes');
  const outcomesEl = document.getElementById('outcomes');
  const transcriptEl = document.getElementById('transcript');

  function setStatus(text, isError) {
    statusEl.textContent = text || '';
    statusEl.className = 'text-sm ' + (isError ? 'text-red-600' : 'text-gray-600');
  }

  function renderOutcomes(items) {
    outcomesEl.innerHTML = '';
    (items || []).forEach((it) => {
      const li = document.createElement('li');
      const owner = it.owner ? ` [Owner: ${it.owner}]` : '';
      const due = it.due ? ` (Due: ${it.due})` : '';
      const prio = it.priority ? ` [Priority: ${it.priority}]` : '';
      li.textContent = `${it.task || ''}${owner}${due}${prio}`;
      outcomesEl.appendChild(li);
    });
  }

  function renderTranscript(utterances) {
    transcriptEl.innerHTML = '';
    (utterances || []).forEach((u) => {
      const div = document.createElement('div');
      div.className = 'p-3 rounded bg-gray-50';
      const startSec = (u.start_ms || 0) / 1000;
      const endSec = (u.end_ms || 0) / 1000;
      const time = `[${startSec.toFixed(1)}s - ${endSec.toFixed(1)}s]`;
      div.textContent = `${u.speaker}: ${u.text} ${time}`;
      transcriptEl.appendChild(div);
    });
  }

  async function handleSubmit(ev) {
    ev.preventDefault();
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus('Please choose a file first.', true);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    submitBtn.disabled = true;
    setStatus('Uploading and processing...');

    try {
      const resp = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await resp.json();
      resultsEl.classList.remove('hidden');
      summaryEl.textContent = data.summary.summary || '';
      notesEl.textContent = data.summary.meeting_notes || '';
      renderOutcomes(data.summary.outcomes || []);
      renderTranscript((data.transcription && data.transcription.utterances) || []);
      setStatus('Done');
    } catch (e) {
      setStatus(e.message || 'Error occurred', true);
    } finally {
      submitBtn.disabled = false;
    }
  }

  form.addEventListener('submit', handleSubmit);
})();
