#include <iostream>


void greet(); //if function is placed after main(), it must be declared in order not to raise an error
void greet_name(std::string user_name);
float square(float side); //data type written before function should match the return data type

void bakePizza();
void bakePizza(std::string topping);
void bakePizza(std::string topping1, std::string topping2);

int main(){

    greet();

    std::string name;

    std::cout<<"Enter your name: ";
    std::cin>>name;

    greet_name(name);

    float length = 10;
    float area;
    area = square(length);
    std::cout<<area<<"\n\n";

    bakePizza();
    bakePizza("Hot Dog");
    bakePizza("Hot Dog", "Mushroom");

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

//Functions Overloading:

void bakePizza(){
    std::cout << "Here is your pizza!\n";
}
void bakePizza(std::string topping1){
    std::cout << "Here is your " << topping1 << " pizza!\n";
}
void bakePizza(std::string topping1, std::string topping2){
    std::cout << "Here is your " << topping1 << " and " << topping2 << " pizza!\n";
}
/*    
While each function share the same name they accept different number of parameters.
The number of parameters passed while calling the function determines which one will run 
*/