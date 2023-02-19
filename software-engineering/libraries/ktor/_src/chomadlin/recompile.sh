# The only way to re-build in heroku is just push
# See https://stackoverflow.com/questions/9713183/recompile-heroku-slug-without-push-or-config-change
git commit --allow-empty -m "[bot] empty commit"
git push heroku master
