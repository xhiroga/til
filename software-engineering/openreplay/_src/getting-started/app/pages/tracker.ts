import Tracker from '@openreplay/tracker';

const { OPENREPLAY_PROJECT_KEY } = process.env;
if (OPENREPLAY_PROJECT_KEY) {
    const tracker = new Tracker({
        projectKey: OPENREPLAY_PROJECT_KEY,
    });
    tracker.start();
}
