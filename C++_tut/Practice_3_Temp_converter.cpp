#include <iostream>

int main(){

    float temp;
    char unit;

    std::cout<<"##########Temp Converter##########\n";
    std::cout<<"Enter Temperature: [Temp] [unit C/F]\n";
    std::cin>>temp>>unit;
    
    switch (unit)
    {
    case 'F':
        std::cout<<temp<<" F ----> ";
        temp = (temp -32) / 1.8;
        std::cout<<temp<<" C";
        break;
    case 'C':
        std::cout<<temp<<" C ----> ";
        temp = (1.8*temp) + 32.0;
        std::cout<<temp<<" F";
        break;
    
    default:
        std::cout<<"Incoreect Input";
        break;
    }

    std::cout<<"##################################\n";

    return 0;
}