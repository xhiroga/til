module.exports = {
    // Example setup for your project:
    // The entry module that requires or imports the rest of your project.
    // Must start with `./`!
    entry: './src/entry.pm.js',
    // Place output files in `./dist/my-app.js`
    target: 'node',
    output: {
      path: __dirname + '/dist',
      filename: 'entry.pm.js'
    }
  };