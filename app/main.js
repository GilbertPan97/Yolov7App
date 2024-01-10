// 引入electron并创建一个Browserwindow
const { app, BrowserWindow, globalShortcut, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

// 保持window对象的全局引用,避免JavaScript对象被垃圾回收时,窗口被自动关闭.
let mainWindow

function createWindow() {
  //创建浏览器窗口,宽高自定义具体大小你开心就好
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    },
  })

  // 加载应用----适用于 react 项目
  mainWindow.loadFile(path.join(__dirname, './dist/index.html'))
}

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

    // 要执行的文件路径
    const executablePath = path.join(__dirname, 'backend', 'ipcBackend');

    // 子进程启动的选项
    const spawnOptions = {
      cwd: path.join(__dirname, 'backend'), // 设置工作目录
    };

    // 通过 child_process.spawn 启动后端可执行文件
    const backendProcess = spawn(executablePath, [], spawnOptions);

    backendProcess.stdout.on('data', (data) => {
      console.log(`Backend stdout: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
      console.error(`Backend stderr: ${data}`);
    });

    backendProcess.on('close', (code) => {
      console.log(`Backend process exited with code ${code}`);
    });

    createWindow()
  })
}

// 实现自定义标题栏，最小化，最大化，关闭
ipcMain.on("window-min", () => mainWindow.minimize());

ipcMain.on("window-max", () => {
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow.maximize();
  }
});

ipcMain.on("window-close", () => {
  mainWindow.destroy();
  backendProcess.kill();
});

