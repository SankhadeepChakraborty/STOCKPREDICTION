from flask import *
import yfinance as yf
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
from flask_mysqldb import MySQL
import random
import smtplib
from email.message import EmailMessage


app = Flask(__name__)

# Secret key for session
app.secret_key = os.urandom(24)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'major'

mysql = MySQL(app)


# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")
    
def fetch_stock_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="5y")
        if stock_data.empty:
            print(f"No data available for {stock_symbol}. Please check the symbol or try again later.")
            return None
        stock_data['Tomorrow'] = stock_data['Close'].shift(-1)
        stock_data['Target'] = (stock_data['Tomorrow'] > stock_data['Close']).astype(int)
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data for {stock_symbol}: {e}")
        return None



def calculate_rsi(stock_data, period=14):
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    return stock_data


def calculate_bollinger_bands(stock_data, window=20):
    stock_data['Middle Band'] = stock_data['Close'].rolling(window).mean()
    stock_data['Upper Band'] = stock_data['Middle Band'] + 2 * stock_data['Close'].rolling(window).std()
    stock_data['Lower Band'] = stock_data['Middle Band'] - 2 * stock_data['Close'].rolling(window).std()
    return stock_data


def calculate_moving_averages(stock_data):
    stock_data['100_MA'] = stock_data['Close'].rolling(window=100).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()
    return stock_data


def calculate_macd(stock_data):
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()  # Signal line calculation
    return stock_data


def plot_dynamic(stock_data, filename, y_data, title):
    fig = go.Figure()
    for y in y_data:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[y], mode='lines', name=y))
    fig.update_layout(title=title, template='plotly_dark')
    filename_html = f"static/{filename}.html"
    fig.write_html(filename_html)
    return filename_html
def plot_accuracy_precision(accuracy, precision):
    fig_acc = go.Figure(go.Pie(
        values=[accuracy, 100 - accuracy],
        labels=['Accuracy', 'Error'],
        hole=0.5,
        marker=dict(colors=['green', 'red'])  # Accuracy in green, Error in red
    ))
    fig_acc.update_layout(paper_bgcolor='black')
    fig_acc.write_html('static/accuracy_chart.html')

    fig_prec = go.Figure(go.Pie(
        values=[precision, 100 - precision],
        labels=['Precision', 'Error'],
        hole=0.5,
        marker=dict(colors=['orange', 'red'])  # Precision in orange, Error in red
    ))
    fig_prec.update_layout(paper_bgcolor='black')
    fig_prec.write_html('static/precision_chart.html')

@app.route("/check_session")
def check_session():
    if "user_id" in session:  # Assuming session stores "user_id"
        return jsonify({"logged_in": True})
    return jsonify({"logged_in": False})



@app.route('/input', methods=['GET', 'POST'])
def input_stock():
    if 'email' not in session:
        flash('You need to log in first!', 'error')
        return redirect('/login')
    return render_template("index2.html")

# Sign-up Route
@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form['email']
        pwd = request.form['pwd']
        name = request.form['name']

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        if cursor.fetchone():
            flash('User already exists!', 'error')
            return redirect('/sign_up')

        cursor.execute('INSERT INTO users (name,email, pwd) VALUES (%s,%s, %s)', (name,email, pwd))
        mysql.connection.commit()
        cursor.close()

        flash('Account created successfully! Please login.', 'success')
        return redirect('/login')

    return render_template('sign_up.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        pwd = request.form['pwd']

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s AND pwd = %s', (email, pwd))
        user = cursor.fetchone()

        if user:
            session['email'] = email
            flash('Login successful!', 'success')
            return redirect('/')
        
        flash('Invalid user or password!', 'error')
    return render_template('login.html')



@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    if request.method == 'POST':
        c_pwd = request.form['current_password']
        n_pwd = request.form['new_password']
        email = request.form['email']
        
        if email is None:
            flash('You must be logged in to change your password.', 'error')
            return redirect('/')
        
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s AND pwd = %s', (email, c_pwd))
        user = cursor.fetchone()

        if user:
            cursor.execute('UPDATE users SET pwd = %s WHERE email = %s', (n_pwd, email))
            mysql.connection.commit()
            cursor.close()
            flash('Password successfully changed!', 'success')
        else:
            flash('Incorrect current password!', 'error')

    return render_template('reset_password.html')
  





@app.route('/logout')
def logout():
    # Remove the 'logged_in' key from the session
    session.pop('logged_in', None)
    # Redirect the user to the homepage
    return redirect(url_for('login'))



@app.route('/about')
def about():
    return render_template('about.html')

    


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'email' not in session:
        flash('You must be logged in to access profile settings.', 'error')
        return redirect('/')

    if request.method == 'POST':
        email = session['email']
        c_pwd = request.form['current_password']
        n_pwd = request.form['new_password']

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s AND pwd = %s', (email, c_pwd))
        user = cursor.fetchone()

        if user:
            cursor.execute('UPDATE users SET pwd = %s WHERE email = %s', (n_pwd, email))
            mysql.connection.commit()
            cursor.close()
            flash('Password successfully changed!', 'success')
        else:
            flash('Incorrect current password!', 'error')

    return render_template('setting.html')  # Stay on the same page!

        
   


def train_model(stock_data):
    stock_data = stock_data.dropna()
    features = ['Close', 'RSI', '100_MA', '200_MA']
    X = stock_data[features]
    y = stock_data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred) * 100, 2)

    plot_accuracy_precision(accuracy, precision)

    latest_data = X.iloc[-1:]
    recommendation = "Hold"

    if latest_data['Close'].values[0] > latest_data['100_MA'].values[0] and latest_data['Close'].values[0] > latest_data['200_MA'].values[0]:
        recommendation = "Buy"
    elif latest_data['Close'].values[0] < latest_data['100_MA'].values[0] or latest_data['Close'].values[0] < latest_data['200_MA'].values[0]:
        recommendation = "Sell"

    return accuracy, precision, recommendation


@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/index", methods=["GET", "POST"])
def index():  # Renamed function
    if 'email' in session:
        stock_symbol = None
        recommendation = None
        last_10_days = None

        if request.method == "POST":
            stock_symbol = request.form['stock_symbol'].strip().upper()
            stock_data = fetch_stock_data(stock_symbol)

            if stock_data is None:
                return render_template('index.html', error="No Data Available for the Stock")

            stock_data = calculate_rsi(stock_data)
            stock_data = calculate_bollinger_bands(stock_data)
            stock_data = calculate_moving_averages(stock_data)
            stock_data = calculate_macd(stock_data)

            plot_dynamic(stock_data, f"{stock_symbol}_ma_100", ['Close', '100_MA'], "100-Day Moving Average")
            plot_dynamic(stock_data, f"{stock_symbol}_ma_200", ['Close', '200_MA'], "200-Day Moving Average")
            plot_dynamic(stock_data, f"{stock_symbol}_rsi", ['RSI'], "Relative Strength Index (RSI)")
            plot_dynamic(stock_data, f"{stock_symbol}_bb", ['Close', 'Upper Band', 'Lower Band'], "Bollinger Bands")
            plot_dynamic(stock_data, f"{stock_symbol}_macd", ['MACD', 'Signal_Line'], "MACD Plot with Signal Line")

            last_10_days = stock_data.tail(10)[['Close', 'RSI', '100_MA', '200_MA']].reset_index()

            accuracy, precision, recommendation = train_model(stock_data)

        if last_10_days is not None:
            last_10_days = last_10_days.to_dict(orient='records')

        return render_template('index.html', stock_symbol=stock_symbol, recommendation=recommendation, last_10_days=last_10_days)

    return redirect("/")



if __name__ == "__main__":
    app.run(debug=True)
