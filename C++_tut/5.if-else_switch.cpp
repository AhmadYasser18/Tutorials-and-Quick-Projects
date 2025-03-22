#include <iostream>

int main(){

    //if statements
    int x=2;
    int y=3;

    if(x>y){
        std::cout<<"X is larger than Y";
         
    }
    else if(y>x){
        std::cout<<"Y is larger than X";
    }
    else{
        std::cout<<"X is equal to Y"<<std::endl;
    }

    //switch statements

    char grade;
    std::cout<<"Enter your Grade";
    std::cin>>grade;

    switch (grade)
    {
    case 'A':
        std::cout<<"Excellent Job";
        break;
    
    case 'B':
        std::cout<<"Good Job";
        break;

    case 'C':
        std::cout<<"Good Job, but try harder";
        break;
    
    default:
        std::cout<<"Enter letters (A - C)";;
    }

    return 0;
}