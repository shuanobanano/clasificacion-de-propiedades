const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('propertyAPI', {
  selectFile: () => ipcRenderer.invoke('file:select'),
  runPrediction: (filePath) => ipcRenderer.invoke('prediction:run', filePath),
});