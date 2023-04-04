from flask import Flask, redirect, url_for, request, render_template, jsonify, flash, session
import pymongo
from flask_bcrypt import Bcrypt
from datetime import datetime

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
app.secret_key = "Jasmine264@51#$23"

bcrypt = Bcrypt(app)

# MongoDB Database
myclient = pymongo.MongoClient("mongodb+srv://amndb:F9fFT4fMiVwE8tre@cluster0.rq92bhz.mongodb.net/?retryWrites=true&w=majority")
mydb = myclient["mydatabase"]

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with home() function.
def home():
    return render_template('index.ejs')

@app.route('/learn')
# ‘/’ URL is bound with learn() function.
def learn():
    return render_template('learn.ejs')

@app.route('/team')
# ‘/’ URL is bound with team() function.
def team():
    return render_template('team.ejs')

@app.route('/contact')
# ‘/’ URL is bound with contact() function.
def contact():
    return render_template('contact.ejs')

@app.route('/login')
def login():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

'''
No need for this route,
need to refactor the authentication process of this route into signup route
'''
@app.route('/register', methods=["GET", "POST"])
def register():
    message = ""
    status = "fail"
    if request.method == 'POST':
        result = request.form
        result = dict(result)
        print(result)
        if(result["Author"] == 'Student'):
            mycol = mydb["student"]
        else:
            mycol = mydb["faculty"]
        try:
            check = mydb["student"].find({"Email": result['Email']})
            check1 = mydb["faculty"].find({"Email": result['Email']})
            if((len(list(check))+len(list(check1))) >= 1):
                message = "User with that Email Exists"
                status = "fail"
            else:
                # hashing the password so it's not stored in the db as it was 
                if(result['Password'] == result['Confirm_Password']):
                    result['Password'] = bcrypt.generate_password_hash(result['Password']).decode('utf-8')
                    del result['Confirm_Password']
                    result['Created'] = datetime.now()
                    print(result)

                    #this is bad practice since the data is not being checked before insert
                    res = mycol.insert_one(result) 
                    if res.acknowledged:
                        status = "successful"
                        message = "User Created Successfully"
                    return redirect(url_for('login'))
                else:
                    message = "Password and Confirm Password do not match"
                    status = "fail"

        except Exception as ex:
            message = f"{ex}"
            status = "fail"
        flash(message)
        return redirect(url_for('signup'))
        #return jsonify({'status': status, "message": message}), 200   


@app.route('/signin', methods = ['GET', 'POST'])
def signin():
    message = ""
    status = "fail"
    if request.method == 'POST':
        session.pop('user', None) 
        result = request.form
        result = dict(result)
        mycol = mydb["student"]
        try:
            user = mycol.find_one({"Email": result['Name']})
            if user:
                print(bcrypt.check_password_hash(user['Password'], result['Password']))
                if(bcrypt.check_password_hash(user['Password'], result['Password'])):
                    message = "User Authenticated"
                    status = "successful"
                    session['user'] = result['Name']
                    return redirect(url_for('exams'))
                else:
                    message = "Wrong Password"
                    status = "fail"
            else:
                message = "Invalid login details"
                status = "fail"

        except Exception as ex:
            message = f"{ex}"
            status = "fail"
        flash(message)
        return redirect(url_for('login'))
        #return jsonify({'status': status, "message":message}), code




# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
