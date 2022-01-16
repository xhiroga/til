const path = require('path');
const appRoot = require('app-root-path');
const winston = require('winston');
const { colorize, combine, timestamp, label, printf } = winston.format;
const options = {
    file: {
        level: 'info',
        filename: `${appRoot}/logs/app.log`,
        handleExceptions: true,
        json: true,
        maxsize: 5242880, // 5MB
        maxFiles: 5,
        colorize: false,
    },
    // Winston is using JSON for logging format by default.
    console: {
        level: 'debug',
        handleExceptions: true,
        json: true,
        colorize: true,
    },
};
const getLabel = function (callingModule) {
    const parts = callingModule.filename.split(path.sep);
    return path.join(parts[parts.length - 2], parts.pop());
};
const getLogger = (callingModule) => {
    const logger = winston.createLogger({
        format: combine(
            label({ label: getLabel(callingModule) }),
            timestamp(),
            colorize(),
            printf(info => {
                return `${info.timestamp} [${info.label}] ${info.level}: ${info.message}`
            })
        ),
        transports: [
            new winston.transports.File(options.file),
            new winston.transports.Console(options.console)
        ],
        exitOnError: false, // do not exit on handled exceptions
    })
    logger.stream = {
        write: function (message, encoding) {
            logger.info(message);
        },
    };
    return logger
};
module.exports = getLogger;
