#include <iostream>

//method 1
typedef std::string text_t; 
typedef int number_t;
//method 2
using decimals = float;  

int main(){

    //normally to use string data type it's used as std::string
    //for other types as well require similar declaration which might get long and complicated 

    text_t name = "Ahmad";
    number_t  score = 95;
    decimals GPA = 2.9;     

    std::cout<<name<<std::endl;
    std::cout<<score<<std::endl;
    std::cout<<GPA<<std::endl;


    return 0;
}