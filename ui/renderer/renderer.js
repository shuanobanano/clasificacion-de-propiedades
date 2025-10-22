// === Helpers numéricos y de formato ===
const isFiniteNumber = (v) => typeof v === 'number' && Number.isFinite(v);

// Convierte strings tipo "USD 500" o "$ 235.000 Expensas" a número; maneja . de miles y , decimal (es-AR)
function toNum(v) {
  if (v == null) return null;
  if (isFiniteNumber(v)) return v;
  if (typeof v === 'string') {
    const cleaned = v
      .replace(/[^\d,.\-]/g, '') // quita símbolos
      .replace(/\.(?=\d{3}(\D|$))/g, '') // saca puntos de miles
      .replace(/,/, '.'); // coma decimal → punto
    const n = Number(cleaned);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

const fmtNumber = (v) => {
  const n = toNum(v);
  return n == null ? '—' : n.toLocaleString('es-AR');
};

const fmtCurrency = (v, currency = 'ARS') => {
  const n = toNum(v);
  return n == null
    ? '—'
    : n.toLocaleString('es-AR', {
        style: 'currency',
        currency,
        maximumFractionDigits: 0,
      });
};

const fmtPercent = (v) => {
  const n = toNum(v);
  return n == null
    ? '—'
    : n.toLocaleString('es-AR', { style: 'percent', maximumFractionDigits: 1 });
};

// Accessor tolerante a distintas formas de objetos (JSON/JSONL/CSV parseado)
function pick(obj, paths, fallback = null) {
  for (const p of paths) {
    const segs = p.split('.');
    let v = obj;
    for (const s of segs) v = v?.[s];
    if (v !== undefined && v !== null) return v;
  }
  return fallback;
}

function classVariant(label) {
  const k = String(label || '').toLowerCase();
  if (k.includes('infravalor')) return 'success';
  if (k.includes('regular')) return 'warning';
  if (k.includes('sobrevalor')) return 'danger';
  if (k.includes('barat')) return 'success';
  if (k.includes('caro')) return 'danger';
  return 'secondary';
}

const selectButton = document.getElementById('select-file');
const processButton = document.getElementById('process');
const fileNameLabel = document.getElementById('file-name');
const statusContainer = document.getElementById('status');
const resultsContainer = document.getElementById('results');

let selectedFile = null;

// Debug: verificar que el API está disponible
console.log('propertyAPI disponible:', !!window.propertyAPI);
console.log('Métodos de propertyAPI:', window.propertyAPI ? Object.keys(window.propertyAPI) : 'No disponible');

function renderStatus(message, type = 'info') {
  statusContainer.textContent = message;
  statusContainer.className = `status ${type}`;
}

function createProbabilityList(probabilities = {}) {
  if (!probabilities || typeof probabilities !== 'object') {
    return '<p>—</p>';
  }

  const normalizeToPercent = (value) => {
    const n = toNum(value);
    if (n == null) return '—';
    const fraction = n > 1 ? n / 100 : n;
    return fmtPercent(fraction);
  };

  const entries = [
    { label: 'BARATO', paths: ['BARATO', 'barato', 'cheap', 'low'] },
    { label: 'REGULAR', paths: ['REGULAR', 'regular', 'mid'] },
    { label: 'CARO', paths: ['CARO', 'caro', 'high', 'expensive'] },
  ];

  const items = entries
    .map(({ label, paths }) => {
      const value = pick(probabilities, paths, null);
      return `<li><strong>${label}:</strong> ${normalizeToPercent(value)}</li>`;
    })
    .join('');

  return `<ul class="probability-list">${items}</ul>`;
}

function renderResults(results = []) {
  if (!Array.isArray(results) || results.length === 0) {
    resultsContainer.innerHTML = '<p>No se generaron predicciones.</p>';
    return;
  }

  const legacyBadgeMap = {
    success: 'barato',
    warning: 'regular',
    danger: 'caro',
  };

  const fragments = [];

  results.forEach((result, index) => {
    if (!result || typeof result !== 'object') {
      return;
    }

    if (result.status === 'error') {
      const errorMessage = pick(result, ['error', 'message'], 'Error desconocido');
      fragments.push(`
        <article class="card error">
          <h3>Propiedad ${index + 1}</h3>
          <p class="error-message">${errorMessage}</p>
        </article>
      `);
      return;
    }

    const location = pick(result, ['input.Location', 'Location', 'location'], '—');
    const price = pick(result, ['input.Price', 'Price', 'price'], null);
    const fair = pick(result, ['pred.pred_fair_price', 'pred_fair_price', 'fair_price'], null);
    const devPct = pick(result, ['pred.deviation_pct', 'deviation_pct'], null);
    const currency = pick(result, ['pred.currency', 'currency'], 'ARS') || 'ARS';
    const label = pick(result, ['pred.class_label', 'class_label', 'prediction'], '—');
    const confidenceRaw = pick(result, ['pred.confidence', 'confidence'], null);
    const pricePerM2 = pick(result, ['pred.price_per_m2', 'price_per_m2'], null);
    const probabilities = pick(result, ['pred.probabilities', 'probabilities'], {}) || {};
    const summary = pick(result, ['ui.summary', 'summary'], null);
    const report = pick(result, ['ui.report', 'report'], null);

    const badgeVariant = classVariant(label);
    const badgeClasses = ['badge', badgeVariant];
    const legacyClass = legacyBadgeMap[badgeVariant];
    if (legacyClass) {
      badgeClasses.push(legacyClass);
    } else if (label && typeof label === 'string') {
      badgeClasses.push(label.toLowerCase());
    }

    const confidenceDisplay = (() => {
      const c = toNum(confidenceRaw);
      if (c == null) return '—';
      const fraction = c > 1 ? c / 100 : c;
      return fmtPercent(fraction);
    })();

    const absDiff = (() => {
      const p = toNum(price);
      const f = toNum(fair);
      if (p == null || f == null) return null;
      return p - f;
    })();

    const priceText = fmtCurrency(price, currency);
    const fairText = fmtCurrency(fair, currency);
    const devText = fmtPercent(devPct);
    const pricePerM2Text = fmtCurrency(pricePerM2, currency);
    const diffText = fmtCurrency(absDiff, currency);

    const reportText = (() => {
      const candidate = report ?? summary;
      if (candidate == null) return '—';
      if (typeof candidate === 'string') return candidate;
      try {
        return JSON.stringify(candidate, null, 2);
      } catch (error) {
        return String(candidate);
      }
    })();

    fragments.push(`
      <article class="card">
        <header>
          <h3>Propiedad ${index + 1}</h3>
          <span class="${badgeClasses.join(' ')}">${label}</span>
        </header>
        <p><strong>Ubicación:</strong> ${location}</p>
        <p><strong>Precio listado:</strong> ${priceText}</p>
        <p><strong>Precio justo estimado:</strong> ${fairText}</p>
        <p><strong>Desviación:</strong> ${devText}</p>
        <p><strong>Diferencia absoluta:</strong> ${diffText}</p>
        <p><strong>Confianza:</strong> ${confidenceDisplay}</p>
        <p><strong>Precio por m²:</strong> ${pricePerM2Text}</p>
        <details>
          <summary>Probabilidades</summary>
          ${createProbabilityList(probabilities)}
        </details>
        <details>
          <summary>Reporte detallado</summary>
          <pre>${reportText}</pre>
        </details>
      </article>
    `);
  });

  if (!fragments.length) {
    resultsContainer.innerHTML = '<p>No se generaron predicciones.</p>';
    return;
  }

  resultsContainer.innerHTML = fragments.join('');
}

selectButton.addEventListener('click', async () => {
  console.log('Botón de selección clickeado');
  
  try {
    if (!window.propertyAPI) {
      throw new Error('propertyAPI no está disponible en window');
    }
    
    if (!window.propertyAPI.selectFile) {
      throw new Error('selectFile no está disponible en propertyAPI');
    }
    
    const filePath = await window.propertyAPI.selectFile();
    console.log('Archivo seleccionado:', filePath);
    
    if (!filePath) {
      console.log('Usuario canceló la selección');
      return;
    }
    
    selectedFile = filePath;
    fileNameLabel.textContent = filePath;
    renderStatus('Archivo listo para procesar', 'info');
    
  } catch (error) {
    console.error('Error al seleccionar archivo:', error);
    renderStatus(`❌ Error: ${error.message}`, 'error');
  }
});

processButton.addEventListener('click', async () => {
  if (!selectedFile) {
    renderStatus('❌ Debe seleccionar un archivo primero.', 'error');
    return;
  }

  renderStatus('Procesando archivo, por favor espere...', 'loading');
  resultsContainer.innerHTML = '';

  try {
    console.log('Iniciando predicción para:', selectedFile);
    const response = await window.propertyAPI.runPrediction(selectedFile);
    console.log('Respuesta recibida:', response);
    
    if (!response) {
      renderStatus('❌ No se recibió respuesta del motor de predicción.', 'error');
      return;
    }

    if (response.status !== 'success') {
      renderStatus(response.error || '❌ Error desconocido durante la predicción', 'error');
      if (response.details) {
        resultsContainer.innerHTML = `<pre>${response.details}</pre>`;
      }
      return;
    }

    renderStatus('✅ Predicciones generadas correctamente.', 'success');
    renderResults(response.results);
  } catch (error) {
    console.error('Error en procesamiento:', error);
    renderStatus(`❌ Error inesperado: ${error.message}`, 'error');
  }
});

// Verificar que los elementos del DOM existen
console.log('Elementos del DOM:');
console.log('- selectButton:', selectButton);
console.log('- processButton:', processButton);
console.log('- fileNameLabel:', fileNameLabel);