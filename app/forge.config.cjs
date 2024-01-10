module.exports = {
  packagerConfig: {
    asar: true,
  },
  rebuildConfig: {},
  makers: [
    {
      name: '@electron-forge/maker-squirrel',
      config: {
        "options": {
          "license": "Apache-2.0"
        }
      },
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin'],
    },
    {
      name: '@electron-forge/maker-deb',
      config: {
        "options": {
          "license": "Apache-2.0"
        }
      },
    },
    {
      name: '@electron-forge/maker-rpm',
      config: {
        "options": {
          "license": "Apache-2.0"
        }
      },
    },
  ],
  plugins: [
    {
      name: '@electron-forge/plugin-auto-unpack-natives',
      config: {
        "options": {
          "license": "Apache-2.0"
        }
      },
    },
  ],
};
