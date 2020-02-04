#include <iostream>
#include <string>

int len(std::string s){
	std::string word = s;
	return(word.size());

}

int main(){
	len("hello world");
}