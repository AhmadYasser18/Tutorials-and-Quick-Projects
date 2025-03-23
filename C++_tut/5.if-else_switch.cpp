#include <iostream>

int main(){

    /*
    Logical Operators:
    - &&    and
    - ||    or
    - !     not
    */
    //if statements
    int x=2;
    int y=3;

    if(x>y){
        std::cout<<"X is larger than Y"<<std::endl<<std::endl;
         
    }
    else if(y>x){
        std::cout<<"Y is larger than X"<<std::endl<<std::endl;
    }
    else{
        std::cout<<"X is equal to Y"<<std::endl<<std::endl;
    }
    /////////////////////////
    //switch statements

    char grade;
    std::cout<<"Enter your Grade";
    std::cin>>grade;

    switch (grade)
    {
    case 'A':
        std::cout<<"Excellent Job"<<std::endl<<std::endl;
        break;
    
    case 'B':
        std::cout<<"Good Job"<<std::endl<<std::endl;
        break;

    case 'C':
        std::cout<<"Good Job, but try harder"<<std::endl<<std::endl;
        break;
    
    default:
        std::cout<<"Enter letters (A - C)"<<std::endl<<std::endl;
    }

    /////////////////////////
    //Ternary Operator ?:
    //Acts as replacement for if/else
    
    x>y ? std::cout<<"X is larger than Y\n\n": std::cout<<"Y is larger than X\n\n";

    int num = 5;
    
    num%2 ? std::cout<<"ODD\n" : std::cout<<"Even\n"; //if 1 then true 0 is false 
    //OR
    std::cout<<(num%2 ? "ODD\n" : "Even\n");
    return 0;
}