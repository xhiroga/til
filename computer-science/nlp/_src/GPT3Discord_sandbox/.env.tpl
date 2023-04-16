OPENAI_TOKEN = op://Personal/OpenAI/KAV_K_GPT_OPENAI_API_KEY
#PINECONE_TOKEN = "<pinecone_token>" # pinecone token if you have it enabled. See readme

# 1. Create Application from https://discord.com/developers/applications
# 2. Bot > Reset Token to generate token
# 3. Disable Bor > Public Bot
# 4. OAuth2 > URL Generator
# See also https://discordpy.readthedocs.io/ja/latest/discord.html
DISCORD_TOKEN = op://Personal/Kav-K_GPT/BOT_TOKEN
DEBUG_GUILD = op://Personal/PrivateServer/SERVER_ID
DEBUG_CHANNEL = op://Personal/PrivateServer/KAV-K_GPT_CHANNEL_ID
ALLOWED_GUILDS = op://Personal/PrivateServer/SERVER_ID
# People with the roles in ADMIN_ROLES can use admin commands like /clear-local, and etc
ADMIN_ROLES = "Admin,Owner"
# People with the roles in DALLE_ROLES can use commands like /dalle draw or /dalle imgoptimize
DALLE_ROLES = "Admin,Openai,Dalle,gpt"
# People with the roles in GPT_ROLES can use commands like /gpt ask or /gpt converse
GPT_ROLES = "openai,gpt"
WELCOME_MESSAGE = "Hi There! Welcome to our Discord server. We hope you'll enjoy our server and we look forward to engaging with you!"  # This is a fallback message if gpt3 fails to generate a welcome message.
USER_INPUT_API_KEYS="False" # If True, users must use their own API keys for OpenAI. If False, the bot will use the API key in the .env file.
# Moderations Service alert channel, this is where moderation alerts will be sent as a default if enabled
MODERATIONS_ALERT_CHANNEL = op://Personal/PrivateServer/KAV-K_GPT_CHANNEL_ID
# User API key db path configuration. This is where the user API keys will be stored.
USER_KEY_DB_PATH = "user_key_db.sqlite"
# Determines if the bot responds to messages that start with a mention of it
BOT_TAGGABLE = "true"

DATA_DIR=/data
SHARE_DIR=/data/share
