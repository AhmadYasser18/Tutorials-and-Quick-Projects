#include <iostream>
using namespace std;

namespace my_space{
    int x = 1;
    int y = 20;
}

namespace space_2{
    int x = 10;
}



int main(){
    using namespace my_space;

    /*
    namespce: provides a solution for preventing name conflicts in large
              projects. Each entity needs a unique name. A namespace allows
              for identically named entities as long as the namespaces are different.
    */

    using std::cout;
    using std::string;
    using std::endl;
    // if using namespace std; wasn't used


    float x = 5; //overrides value of x from my_space; however won't raise error as x still can accessed using my_space::x

    cout<<"From main function: "<<x<<endl;
    cout<<"From my_space: "<<my_space::x<<endl;
    cout<<"From my_space: "<<y<<endl;
    cout<<"From space_2: "<<space_2::x<<endl;

     
    return 0;
}