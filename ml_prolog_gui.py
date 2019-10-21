import os
import tkinter as tk
from tkinter import ttk
import subprocess
from tkinter import messagebox
import MLpredict as ml
import ML_random_forests as rf

# !!! Works only on Windows looks for swipl on C:/ drive, if swi prolog is installed elsewhere
# modify C:/ to the correct partition/hard disk
# This will slow down the program but should increase portability for testing


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


swipl = find("swipl.exe", "C:/")
print("---")
swipl = swipl.split('.')
swipl = swipl[0]
swipl = swipl.replace("\\", "/")
print(swipl)
print("---")

w = tk.Tk()
w.title('Risk Assessment')
tk.Label(w, text="First Name").grid(row=0)
tk.Label(w, text="Last Name").grid(row=1)
tk.Label(w, text="Age").grid(row=2)
tk.Label(w, text="Country").grid(row=3)
tk.Label(w, text="BMI").grid(row=0, column = 2)
tk.Label(w, text="Surgeries").grid(row=1, column = 2)
tk.Label(w, text="Docvists").grid(row=2, column = 2)

name_entry = tk.Entry(w)
surname_entry = tk.Entry(w)
age_entry = tk.Entry(w)
bmi_entry = tk.Entry(w)
surgeries_entry = tk.Entry(w)
docvisits_entry = tk.Entry(w)



name_entry.grid(row=0, column=1)
surname_entry.grid(row=1, column=1)
age_entry.grid(row=2, column = 1)
bmi_entry.grid(row=0, column=3)
surgeries_entry.grid(row=1, column=3)
docvisits_entry.grid(row=2, column=3)

#Check how place of residence is important

OPTIONS = []
with open("./places", 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.split("\t")[0]
        OPTIONS.append(line)

variable = tk.StringVar(w)
variable.set(OPTIONS[0])
option_ailment = tk.OptionMenu(w, variable, *OPTIONS).grid(row=3, column = 1)

var1 = tk.IntVar()
chkbtn1 = ttk.Checkbutton(w, text ='hearing',
             takefocus = 0, variable = var1)
chkbtn1.grid(row=4,column= 0)
var2 = tk.IntVar()
chkbtn2 = ttk.Checkbutton(w, text ='mental',
             takefocus = 0, variable=var2)
chkbtn2.grid(row=4,column= 1)

var3 = tk.IntVar()
chkbtn3 = ttk.Checkbutton(w, text ='cholesterol',
             takefocus = 0, variable=var3)
chkbtn3.grid(row=4,column= 2)

var4 = tk.IntVar()
chkbtn4 = ttk.Checkbutton(w, text ='diabetes',
             takefocus = 0, variable=var4)
chkbtn4.grid(row=5,column=0)

var5 = tk.IntVar()
chkbtn5 = ttk.Checkbutton(w, text ='heart',
             takefocus = 0, variable = var5)
chkbtn5.grid(row=5,column= 1)

var6 = tk.IntVar()
chkbtn6 = ttk.Checkbutton(w, text ='allergies',
             takefocus = 0, variable=var6)
chkbtn6.grid(row=5,column= 2)

var7 = tk.IntVar()
chkbtn7 = ttk.Checkbutton(w, text ='other',
             takefocus = 0, variable= var7)
chkbtn7.grid(row=5,column= 3)
var8 = tk.IntVar()
chkbtn8 = ttk.Checkbutton(w, text ='movement',
             takefocus = 0, variable=var8)
chkbtn8.grid(row=4,column= 3)

var9 = tk.IntVar()
chkbtn9 = ttk.Checkbutton(w, text ='Pay to reduce risk if medium?',
             takefocus = 0, variable=var9)
chkbtn9.grid(row=6,column=1)

var10 = tk.IntVar()
chkbtn10 = ttk.Checkbutton(w, text ='Does the client take medications?',
             takefocus = 0, variable=var10)
chkbtn10.grid(row=7,column=1)


def check_age():
    try:
        age_int = int(age_entry.get())
        if 20 < age_int < 71:
            print(age_int)
            print("age ok")
            global accept
            accept += 1
        else:
            print("age not within range")
    except ValueError:
        print("Invalid Input")

# Retrieve all entries and prepare queries
def evaluate():
    age_int = age_entry.get()
    country = variable.get()
    bmi = bmi_entry.get()
    surgeries = surgeries_entry.get()
    docvisits = docvisits_entry.get()
    allergies = var6.get()
    medication = var10.get()
    cholesterol = var3.get()
    diabetes = var4.get()
    heart = var5.get()
    # Prepare tensors for ML models
    try:
        prediction_tensor = ml.make_prediction_tensor(age_int,surgeries,docvisits,allergies,medication,cholesterol,diabetes,heart,bmi)
        rank, probability = ml.predict(prediction_tensor)
        random_forest_pred = rf.make_prediction_array(age_int,surgeries,docvisits,allergies,medication,cholesterol,diabetes,heart,bmi)
        prediction_rf = rf.predict(random_forest_pred)
        print(rank)
        print(probability)
    except:
        print("insert all values")

    diseases = []
    diseases.append(var1.get())
    diseases.append(var2.get())
    diseases.append(var3.get())
    diseases.append(var4.get())
    diseases.append(var5.get())
    diseases.append(var6.get())
    diseases.append(var7.get())
    diseases.append(var8.get())
    chkbtnList =[]
    chkbtnList.append(chkbtn1.cget("text"))
    chkbtnList.append(chkbtn2.cget("text"))
    chkbtnList.append(chkbtn3.cget("text"))
    chkbtnList.append(chkbtn4.cget("text"))
    chkbtnList.append(chkbtn5.cget("text"))
    chkbtnList.append(chkbtn6.cget("text"))
    chkbtnList.append(chkbtn7.cget("text"))
    chkbtnList.append(chkbtn8.cget("text"))
    print(chkbtnList)
    print(diseases)
    queryDiseases = []
    counter = 0
    # If a disease is checked, add it to the list of diseases (for prolog query)
    for i in diseases:
        print(diseases[i])
        if diseases[counter] == 1:
            queryDiseases.append(chkbtnList[counter])
        counter = counter + 1

    print(queryDiseases)
    reduce = chkbtn9.cget("text")
    if reduce == 1:
        pay_to_reduce = "yes"
    else:
        pay_to_reduce = "no"
    query_to_disk = ("findRisk(" + age_int + "," + "Risk," + str(queryDiseases) + "," + country + "," + pay_to_reduce + ").")
    query_file = open("query.txt", "w")
    query_file.write(query_to_disk)
    query_file.close()
    process = subprocess.call('\"{}\" -f prolog_code.pl < query.txt > output.txt'.format(swipl),
                              shell=True)
    f = open("output.txt", "r")
    risk = f.read()
    # Remove empty line
    risk = os.linesep.join([s for s in risk.splitlines() if s])

    if risk == "Risk = rejected.":
        reccomended = "Expert System"
    elif risk == "false.":
        reccomended = "Expert System, candidate ineligible"
    else:
        reccomended = "Tensorflow"
    try:
        messagebox.showinfo("Risk Evaluation", "Expert System (prolog): "+risk+"\n"+"Tensorflow: " + str(rank) + "({:4.1f}%)".format(probability) +  "\n"+"(Testing) Random Forests: " + prediction_rf + "\n Reccomended: " + reccomended)
    except:
        messagebox.showinfo("Risk Evaluation", "Expert System (prolog): "+risk)

    print(risk)
    print(query_to_disk)
    f.close()


def show_entry_fields():
    print("First Name: " + (name_entry.get()))



tk.Button(w,
          text='Assess Risk',  command=evaluate).grid(row=9, column= 1)


w.mainloop()
