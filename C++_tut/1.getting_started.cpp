#include <iostream>
#include <string>
using namespace std;

/* main() function: where the execution of
C++ program begins */
int main() {
  
    // This statement prints "Hello World"
    
    cout << "Hello Ahmad"<<endl;

    ///////////////////////
    /*Variables:
    There are two steps to create a ariable: declaration and assignment 
    */
   //integer (whole numbers)
    int a;
    a = 10;

    int b = 5;

    cout<<a<<endl<<b<<endl;
    
    int c;
    c = 7.5; 
    cout<<c; //prints as 7

    //double (number including decimal)
    double d;
    d = 7.5;

    cout<<' '<<d<<endl; //prints as 7.5

    //char (single character)
    char letter = 'a';

    cout<<letter<<endl;

    //boolean (true/false)

    bool active = true;

    cout<<"Is active: "<<active<<endl;

    /*
    string: - object representing sequence of text; 
            - provided from 'std' namespace; 
            - if not used then -> std::string var
    */

    string name = "Ahmad";

    cout<<"Hello "<<name<<endl;
    ///////////////////////////////
    const int pi= 3.14; //returns error in case variable is to be changed later 

    //////////////////////////////
    //Operators

    int sum = a +b; //15

    sum = sum + 1; //adds one
    sum+= 1; //adds one
    sum ++; //adds one

    int remainder = sum%4;
    cout<<sum<<endl<<remainder<<endl;

    //Order: parenthesis > multiplication or division > addition or subtraction
    //////////////////////////////
    /* Type conversion : conversion of one data type ro another
                        Implicit: automatic
                        Explicit: precede value with data type
    */

    int x1 = 3.14; //changes to 3 as int only accepts whole numbers (implicit)

    double x2 = (int) 3.14; //explicit

    cout<<"x1 "<<x1<<"_x2 "<<x2<<endl;

    x1 = 2.5; //returns 2
    x2 = 2.5; //returns 2.5 as type is double
    cout<<"x1 "<<x1<<"_x2 "<<x2<<endl;

    char x3 = 100; //returns ASCII 100

    //when dividing integers:
    int m1 = 2;
    int m2 = 4;
    
    cout<<m1/m2; //return 0
    cout<< double(m1)/m2<<endl; //returns 0.5
    //////////////////////////////
    //Taking input from user
    
    cout<<"What is your name? ";
    cin>>name;
    cout<<"Hello "<<name<<endl;

    cout<<"What is your full name? ";
    cin>>name; //If "A Y" is entered
    cout<<"Hello "<<name<<endl; //"Hello A" is returned

    //to get full line
    cout<<"What is your full name? ";
    
    cin.ignore();  // Ignores leftover newline
    cin.clear();

    std::getline(std::cin, name); 
    cout<<"Hello "<<name<<endl; //"Hello A Y" is returned

    return 0;
}