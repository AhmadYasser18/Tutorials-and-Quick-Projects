#include <iostream>


void greet(); //if function is placed after main(), it must be declared in order not to raise an error
void greet_name(std::string user_name);
float square(float side); //data type written before function should match the return data type

int main(){

    greet();

    std::string name;

    std::cout<<"Enter your name: ";
    std::cin>>name;

    greet_name(name);

    float length = 10;
    float area;
    area = square(length);
    std::cout<<area;

    return 0;
}


void greet(){  //function name
    std::cout<<"Hello User!\n";
}

void greet_name(std::string user_name){  //if an argument is required, its type must be added as well 
    std::cout<<"Hello "<<user_name<<"!\n";
}

float square(float side){
    return side*side;
}