// main.js
const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// === NUEVO: cargar config.json con defaults seguros
function loadConfig() {
  const cfgPath = path.join(__dirname, 'config.json');
  const defaults = {
    app: {
      indexHtml: 'renderer/index.html',
      preloadJs: 'preload.js',
      openDevTools: process.env.NODE_ENV === 'development',
      fileFilters: [{ name: 'Datos de propiedades', extensions: ['json', 'jsonl', 'csv', 'joblib', 'pkl', 'md', 'markdown'] }]
    },
    python: {
      executable: process.platform === 'win32' ? 'python' : 'python3',
      argsTemplate: ['-m', 'src.predict', '--input', '${file}', '--jsonl', '--band_pct', '0.10'],
      workingDir: path.join(__dirname, '..', '..')
    },
    spawn: {
      output: 'jsonl' // 'json' si tu backend devuelve un único JSON
    }
  };

  try {
    const raw = fs.readFileSync(cfgPath, 'utf-8');
    const user = JSON.parse(raw);
    // merge superficial (simple y suficiente)
    return {
      app: { ...defaults.app, ...(user.app || {}) },
      python: { ...defaults.python, ...(user.python || {}) },
      spawn: { ...defaults.spawn, ...(user.spawn || {}) }
    };
  } catch {
    return defaults;
  }
}

const config = loadConfig();

// mantener variable para devtools
const isDevelopment = !!config.app.openDevTools;

// preferir config.python.executable pero permitir override por env
const getPythonExecutable = () => {
  if (process.env.PYTHON_EXECUTABLE) return process.env.PYTHON_EXECUTABLE;
  return config.python.executable;
};

const createWindow = () => {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: 'Clasificador de Propiedades',
    webPreferences: {
      preload: path.resolve(__dirname, config.app.preloadJs),
      contextIsolation: true,
      enableRemoteModule: false,
      nodeIntegration: false
    },
  });

  // === NUEVO: indexHtml desde config
  const indexPath = path.resolve(__dirname, config.app.indexHtml);
  mainWindow.loadFile(indexPath);

  if (isDevelopment) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
  return mainWindow;
};

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// === NUEVO: filtros de archivo desde config.json
ipcMain.handle('file:select', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: config.app.fileFilters
  });
  if (canceled || filePaths.length === 0) return null;
  return filePaths[0];
});

// === NUEVO: argsTemplate/cwd/output parsing desde config.json
ipcMain.handle('prediction:run', async (_event, filePath) => {
  if (!filePath) {
    return { status: 'error', error: '❌ Error: Debe seleccionar un archivo para procesar' };
  }

  const pythonExecutable = getPythonExecutable();

  // Construir args a partir de la plantilla (reemplaza ${file})
  const args = (config.python.argsTemplate || []).map(tok =>
    typeof tok === 'string' ? tok.replace('${file}', filePath) : tok
  );

  const cwd = path.isAbsolute(config.python.workingDir)
    ? config.python.workingDir
    : path.resolve(__dirname, config.python.workingDir);

  return new Promise((resolve) => {
    const child = spawn(pythonExecutable, args, {
      cwd,
      env: { ...process.env },
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

    child.on('error', (error) => {
      resolve({ status: 'error', error: `❌ Error al ejecutar Python: ${error.message}` });
    });

    child.on('close', (code) => {
      if (code !== 0 || stderr.trim()) {
        resolve({ status: 'error', error: stderr.trim() || `Proceso Python terminó con código ${code}` });
        return;
      }

      try {
        if (config.spawn.output === 'jsonl') {
          // Parsear JSONL → array de objetos
          const lines = stdout.split(/\r?\n/).filter(Boolean);
          const results = lines.map(l => JSON.parse(l));
          resolve({ status: 'success', results });
        } else {
          // Un único JSON
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        }
      } catch (error) {
        resolve({
          status: 'error',
          error: '❌ Error: No se pudo interpretar la respuesta del modelo',
          details: stdout
        });
      }
    });
  });
});
