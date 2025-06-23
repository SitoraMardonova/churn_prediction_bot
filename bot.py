import logging
import pandas as pd
import joblib
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    ConversationHandler, ContextTypes
)


(TENURE, CONTRACT, INTERNET, MONTHLY, GENDER, PARTNER, DEPENDENTS, SECURITY, TECHSUPPORT, STREAMING, PAYMENT) = range(11)


model = joblib.load("churn_model_1.pkl")
scaler = joblib.load("scaler.pkl")


logging.basicConfig(level=logging.INFO)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Salom! /predict buyrug‘i orqali mijoz holatini bashorat qilamiz.")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("1. Mijozning kompaniyada qolgan oy soni (tenure)?")
    return TENURE

async def get_tenure(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tenure = float(update.message.text.strip())
        context.user_data['tenure'] = tenure
        reply = ReplyKeyboardMarkup([['Month-to-month'], ['One year'], ['Two year']], one_time_keyboard=True)
        await update.message.reply_text("2. Shartnoma turi (Contract)?", reply_markup=reply)
        return CONTRACT
    except ValueError:
        await update.message.reply_text("❌ Iltimos, son kiriting.")
        return TENURE

async def get_contract(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['Contract'] = update.message.text
    reply = ReplyKeyboardMarkup([['DSL'], ['Fiber optic'], ['No']], one_time_keyboard=True)
    await update.message.reply_text("3. InternetService turi?", reply_markup=reply)
    return INTERNET

async def get_internet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['InternetService'] = update.message.text
    await update.message.reply_text("4. Oylik to‘lov (MonthlyCharges)?")
    return MONTHLY

async def get_monthly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        monthly = float(update.message.text.strip().replace(",", "."))
        context.user_data['MonthlyCharges'] = monthly
        await update.message.reply_text("5. Jinsi (gender)?", reply_markup=ReplyKeyboardMarkup([["Male"], ["Female"]], one_time_keyboard=True))
        return GENDER
    except ValueError:
        await update.message.reply_text("❌ Raqam noto‘g‘ri. Qaytadan kiriting.")
        return MONTHLY

async def get_gender(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['gender'] = update.message.text
    reply = ReplyKeyboardMarkup([["Yes"], ["No"]], one_time_keyboard=True)
    await update.message.reply_text("6. Uylanganmi/yashaydigan jufti bormi? (Partner)", reply_markup=reply)
    return PARTNER

async def get_partner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['Partner'] = update.message.text
    reply = ReplyKeyboardMarkup([["Yes"], ["No"]], one_time_keyboard=True)
    await update.message.reply_text("7. Bog‘liqlari (farzand, qarindosh) bormi? (Dependents)", reply_markup=reply)
    return DEPENDENTS

async def get_dependents(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['Dependents'] = update.message.text
    reply = ReplyKeyboardMarkup([["Yes"], ["No"]], one_time_keyboard=True)
    await update.message.reply_text("8. Online xavfsizlik xizmati (OnlineSecurity) bormi?", reply_markup=reply)
    return SECURITY

async def get_security(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['OnlineSecurity'] = update.message.text
    reply = ReplyKeyboardMarkup([["Yes"], ["No"]], one_time_keyboard=True)
    await update.message.reply_text("9. Texnik yordam xizmati (TechSupport) bormi?", reply_markup=reply)
    return TECHSUPPORT

async def get_techsupport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['TechSupport'] = update.message.text
    reply = ReplyKeyboardMarkup([["Yes"], ["No"]], one_time_keyboard=True)
    await update.message.reply_text("10. Streaming TV xizmati bormi? (StreamingTV)", reply_markup=reply)
    return STREAMING

async def get_streaming(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['StreamingTV'] = update.message.text
    reply = ReplyKeyboardMarkup([["Electronic check"], ["Mailed check"], ["Bank transfer (automatic)"], ["Credit card (automatic)"]], one_time_keyboard=True)
    await update.message.reply_text("11. To‘lov usuli (PaymentMethod)?", reply_markup=reply)
    return PAYMENT

async def get_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['PaymentMethod'] = update.message.text
    df = pd.DataFrame([context.user_data])
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

    df = pd.get_dummies(df)
    for col in model.feature_names_in_:
        if col not in df:
            df[col] = 0
    df = df[model.feature_names_in_]

    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    if pred == 1:
        msg = f"⚠️ Mijoz ketishi mumkin. Ehtimol: {proba*100:.1f}%"
    else:
        msg = f"✅ Mijoz ketmaydi. Ehtimol: {(1-proba)*100:.1f}%"

    await update.message.reply_text(msg)
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bekor qilindi.")
    return ConversationHandler.END

def main():
    app = ApplicationBuilder().token("7227715692:AAGYXJiY0qjUkrGavMuk2EzFTvaMYhg39Fs").build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("predict", predict)],
        states={
            TENURE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_tenure)],
            CONTRACT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_contract)],
            INTERNET: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_internet)],
            MONTHLY: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_monthly)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_gender)],
            PARTNER: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_partner)],
            DEPENDENTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_dependents)],
            SECURITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_security)],
            TECHSUPPORT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_techsupport)],
            STREAMING: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_streaming)],
            PAYMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_payment)],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)
    logging.info("🤖 Bot muvaffaqiyatli ishga tushdi. Endi siz undan foydalanishingiz mumkin.")
    app.run_polling()

if __name__ == "__main__":
    main()
