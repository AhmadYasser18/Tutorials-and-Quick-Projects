#SOLOLEARN Longest Word Finder
"""

from tkinter import *

main_window=Tk()
main_window.title("SOLOLEARN")
Label(main_window,text="Longest Word Finder").grid(row=0,column=1)

sentence=Entry(main_window, width=40, borderwidth=10)
sentence.grid(row=1,column=1)

#GUI Main Function
def findlongest():
    y=sentence.get()
    y=y.split()
    length=0
    word=""
    for i in y:
        if len(i)>length:
            length=len(i)
            word=i
    print(word,length)


button_widget1=Button(main_window,text="Done",command=findlongest).grid(row=2,column=1)
button_widget2=Button(main_window,text="End",width=20).grid(row=3,column=2)

main_window.mainloop()
"""
from tkinter import *
import numpy as np


def gameon():

    main_window=Tk()
    main_window.title("SOLOLEARN")
    Label(main_window,text="Tic Tac Toe").grid(row=0,column=2,pady=15,padx=30)


    #GUI Main Function

    d=[1,2,3,4,5,6,7,8,9]
    def checking():

        for i in range(3):
            x=[0,3,6]
            y=[0,1,2]
            if d[0+x[i]]==d[1+x[i]]==d[2+x[i]]:
                if d[0+x[i]]=="X":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Won!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break
                elif d[0+x[i]]=="O":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Lost!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break
            if d[0+y[i]]==d[3+y[i]]==d[6+y[i]]:
                if d[0+y[i]]=="X":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Won!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break
                elif d[0+y[i]]=="O":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Lost!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break
            if d[2]==d[4]==d[6]:
                if d[4]=="X":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Won!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break
                elif d[4]=="O":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Lost!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break
            if d[0]==d[4]==d[8]:
                if d[4]=="X":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Won!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break
                elif d[4]=="O":
                    print("Game Over")
                    Label(main_window,text="Game Over!\nYou Lost!").grid(row=6,column=2,pady=15,padx=30)
                    button_again=Button(main_window,text="Play Again",width=30,height=2.5,command=gameon).grid(row=7,column=2,pady=15,padx=30)
                    break


    def computer_move():
        move=np.random.randint(1,10)
        while move not in d:
            #print(move in d)

            move=np.random.randint(1,10)

        d[move-1]= "O"
        #print(d,move)
        r=2+int(move//3)
        if move%3==0:
            r-=1
        c=move%3
        if c==0:
            c=3
        #print(r in [2,3,4])
        #print(c in [1,2,3])
        button_n=Button(main_window,text="O",width=20,height=10).grid(row=r,column=c)
        checking()



    def removing(n):
        d[n-1]="X"
        r=2+int(n//3)
        if n%3==0:
            r-=1
        c=n%3
        if c==0:
            c=3
        button_n=Button(main_window,text="X",width=20,height=10).grid(row=r,column=c)
        checking()
        computer_move()


    button_1=Button(main_window,text="1",width=20,height=10,command=lambda : removing(1)).grid(row=2,column=1)
    button_2=Button(main_window,text="2",width=20,height=10,command=lambda : removing(2)).grid(row=2,column=2)
    button_3=Button(main_window,text="3",width=20,height=10,command=lambda: removing(3)).grid(row=2,column=3)

    button_4=Button(main_window,text="4",width=20,height=10,command=lambda:removing(4)).grid(row=3,column=1)
    button_5=Button(main_window,text="5",width=20,height=10,command=lambda:removing(5)).grid(row=3,column=2)
    button_6=Button(main_window,text="6",width=20,height=10,command=lambda:removing(6)).grid(row=3,column=3)

    button_7=Button(main_window,text="7",width=20,height=10,command=lambda:removing(7)).grid(row=4,column=1)
    button_8=Button(main_window,text="8",width=20,height=10,command=lambda:removing(8)).grid(row=4,column=2)
    button_9=Button(main_window,text="9",width=20,height=10,command=lambda:removing(9)).grid(row=4,column=3)

    main_window.mainloop()
gameon()
