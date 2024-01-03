const initialVideoId = 'iA91WPotG2I'; // ここに初期動画のIDを設定

// YouTubeプレイヤーを生成するための関数
function onYouTubeIframeAPIReady() {
    player = new YT.Player('player', {
        height: '360',
        width: '640',
        videoId: initialVideoId,
        events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange
        }
    });
}

// プレイヤーが準備完了したときのイベント
function onPlayerReady(event) {
    updateVideoInfo();
}

// プレイヤーの状態が変化したときのイベント
function onPlayerStateChange(event) {
    if (event.data == YT.PlayerState.PLAYING) {
        updateVideoInfo();
    }
}

// 動画情報を更新する関数
function updateVideoInfo() {
    var videoData = player.getVideoData();
    document.getElementById('currentVideoTitle').innerText = 'タイトル: ' + videoData.title;
    var duration = player.getDuration();
    document.getElementById('currentVideoDuration').innerText = '再生時間: ' + formatTime(duration);
}

// 秒を時間形式に変換する関数
function formatTime(seconds) {
    var min = Math.floor(seconds / 60);
    var sec = seconds % 60;
    return min + ':' + (sec < 10 ? '0' : '') + sec;
}

// 次の動画を読み込む関数
function loadNextVideo() {
    var nextVideoId = document.getElementById('nextVideoId').value;
    if (nextVideoId) {
        player.loadVideoById(nextVideoId);
    }
}
