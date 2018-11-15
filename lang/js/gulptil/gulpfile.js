function defaultTask(cb) {
    // place code for your default task here
    console.log('cb:', cb.name) // done
    cb();
}

exports.default = defaultTask