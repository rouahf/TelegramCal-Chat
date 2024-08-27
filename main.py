from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google.generativeai import generate_text, configure

# Configure the API key
API_KEY: Final = 'AIzaSyAD35w9sxvo7DTnL85e0F5BsBsTH60g8xY'
configure(api_key=API_KEY)

# Constants
TOKEN: Final = '7023131652:AAFForSIwVwVcZotWFY6L3VHJ_NGSq0Lmhg'
BOT_USERNAME: Final = '@calendar_calendar_bot'

# Command Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me! I am your calendar Bot.')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('I am a calendar bot! Please type something so I can respond.')

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('This is a custom command!')

# Message Response Handler
def generate_response(text: str) -> str:
    try:
        # Generate text based on the input text
        response = generate_text(prompt=text)

        # Extract and return the response text
        if response and hasattr(response, 'result'):
            return response.result
        else:
            return 'Sorry, I could not generate a response for that. Please try something else.'
    except Exception as e:
        return f'An error occurred: {str(e)}'

# Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    # Generate the response based on the message text
    response: str = generate_response(text)
    
    # Print the response for debugging
    print('Bot:', response)
    
    try:
        # Send the response to Telegram
        await update.message.reply_text(response)
    except Exception as e:
        print(f'Failed to send message: {e}')

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print('Starting bot ...')
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))
    
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    app.add_error_handler(error)
   
    print('Polling...')
    app.run_polling(poll_interval=3)
