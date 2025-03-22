#include <iostream>
#include <cmath>

int main(){
    double a,b,c;

    std::cout<<"Enter value for a: ";
    std::cin>>a;

    std::cout<<"Enter value for b: ";
    std::cin>>b;

    c = sqrt(pow(a,2) + pow(b,2));
    std::cout<<"Hypotenuse value: "<<c;

    return 0;
}