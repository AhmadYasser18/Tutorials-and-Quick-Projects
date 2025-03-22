#include <iostream>

int main(){

    int x, y;
    char op;

    std::cout<<"Enter in the following format:\nX + Y\nWhere X and Y are numbers and + could be [+ - * /]\nMake sure to leave spaces."<<std::endl;

    std::cout<<"\nEnter operation: ";
    std::cin>>x>>op>>y;

    switch (op)
    {
    case '+':
        std::cout<<x+y;
        break;
    case '-':
        std::cout<<x-y;
        break;

    case '*':
        std::cout<<x*y;
        break;

    case '/':
        std::cout<<double(x)/y;
        break;
    default:
        std::cout<<"Enter operator from (- +  /)";
        break;
    }

    return 0;
}