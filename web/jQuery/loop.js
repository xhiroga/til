$(function(){
    setInterval(function(){
        $('#banana').stop().animate({width:'+=' + '10' + 'px'}, 300);
        // .stop() -> 現在かかっている他のアニメーションを中止
        // .animate() -> 書き換えるプロパティと、どのくらいの長さの時間アニメーションするかを指定
    },1000);
});
