#include <iostream>

int main(){

    std::string username;
    std::string mail;
    char confirm;

    std::cout<<"Enter Username: ";
    std::getline(std::cin, username);

    if (username.empty()){
        std::cout<<"You didn't enter anything!";
    }
    else if (username.length()< 8){
        std::cout<<"Username should be made from more than 8 characters.";
    }
    else{
        std::cout<<"To confirm press 1: ";
        std::cin>>confirm;
        if (confirm == '1'){
            std::cout<<"Saved!\n"<<"Your mail is: "<<username.append("@gmail.com")<<std::endl;

            std::cout<<"Your password is: "<<username.at(0)<<username.at(username.length()%5)<<username.at(7)<<std::endl;
            
            mail = username;
            mail.insert(mail.length(), "@gmail.com"); //adds string at given index
            mail.erase(2,5); //erases between given indices

            std::cout<<mail<<std::endl;
            std::cout<<mail.find('A'); //returns index
        }
        else{
            username.clear(); //clears variable content
            std::cout<<"Erased!";
        }
    }


    return 0;
}