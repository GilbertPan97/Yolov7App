// 引入electron并创建一个Browserwindow
const { app, BrowserWindow, globalShortcut, ipcMain } = require('electron')
const path = require('path')
// 保持window对象的全局引用,避免JavaScript对象被垃圾回收时,窗口被自动关闭.
let mainWindow

function createWindow() {
  //创建浏览器窗口,宽高自定义具体大小你开心就好
  mainWindow = new BrowserWindow({
    frame: false,
    fullscreen: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    },
  })
  // 加载应用----适用于 react 项目
  mainWindow.loadFile(path.join(__dirname, './dist/index.html'))
}


ipcMain.on('closed', function () {
  mainWindow.destroy() // 关闭
})

ipcMain.on('window-min', function () {
  mainWindow.minimize()// 最小化
})

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})
const isFirstInstance = app.requestSingleInstanceLock()
if (!isFirstInstance) {
  console.log('is second instance')
  setTimeout(() => {
    app.quit()
  }, 30000)
} else {
  app.on('ready', () => {
    globalShortcut.register('f5', function () {
      console.log('f5 is pressed')
      //mainWindow.reload()
    })
    globalShortcut.register('CommandOrControl+R', function () {
      console.log('CommandOrControl+R is pressed')
      //mainWindow.reload()
    })
    createWindow()
  })
}

// 实现自定义标题栏，最小化，最大化，关闭
// ipcMain.on("window-min", () => win.minimize());
// ipcMain.on("window-max", () => {
//   if (win.isMaximized()) {
//     win.unmaximize();
//   } else {
//     win.maximize();
//   }
// });
// ipcMain.on("window-close", () => {
//   win.destroy();
// });

