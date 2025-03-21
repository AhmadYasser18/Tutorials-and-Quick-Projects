#include <iostream>
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
    
    return 0;
}