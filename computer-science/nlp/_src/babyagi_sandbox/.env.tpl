# cp .env.example .env
# Edit your .env file with your own values
# Don't commit your .env file to git/push to GitHub!
# Don't modify/delete .env.example unless adding extensions to the project
# which require new variable to be added to the .env file

# API CONFIG
OPENAI_API_KEY=op://Personal/OpenAI/BABYAGI
OPENAI_API_MODEL=gpt-3.5-turbo # alternatively, gpt-4, text-davinci-003, etc
OPENAI_TEMPERATURE=0.0
PINECONE_API_KEY=op://Personal/Pinecone/babyagi
PINECONE_ENVIRONMENT=northamerica-northeast1-gcp

# TABLE CONFIG
TABLE_NAME=baby-agi-test-table

# INSTANCE CONFIG
BABY_NAME=BabyAGI

# RUN CONFIG
OBJECTIVE=Solve world hunger
# For backwards compatibility
# FIRST_TASK can be used instead of INITIAL_TASK
INITIAL_TASK=Develop a task list

# Extensions
# List additional extensions to load (except .env.example!)
DOTENV_EXTENSIONS=
# Set to true to enable command line args support
ENABLE_COMMAND_LINE_ARGS=false
