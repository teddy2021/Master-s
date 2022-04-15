// Kody Manastyrski
// CP 83166
// Computational Geometry
// Dialated Number cpp
//

#ifndef dialate
#define dialate
#include "DialatedNumber.hpp"
#include <math.h>
#include<bitset>

using namespace std;

int toDialated(int val){
	char v = (val < 0)? '1':'0';
	string param = bitset<64>(val).to_string();
	string binary = "";
	for(int i = 0; i < param.length() - 1; i += 1){
		char digit = param[i];
		binary.append(&digit);
		binary.append(&v);
	}
	binary += param[param.length() - 1];
	return (int)bitset<64>(binary).to_ullong();
}

Dialated::Dialated(int val){
	value = toDialated(val);
	mask = bitset<64>(-1);
}

int Dialated::to_int(){
	string binary = bitset<64>(value).to_string();
	string out = "";
	while(binary.size() > 0){
		out.append(&binary[0]);
		binary.erase(0,2);
	}
	return (int)bitset<64>(out).to_ullong();
	
}

Dialated Dialated::operator+(const Dialated& other){
	return value + other.value;
}

Dialated Dialated::operator-(){
	return -value;
}

Dialated Dialated::operator-(const Dialated& other){
	return value - other.value;
}

bool Dialated::operator==(Dialated& other){
	return value == other.value;
}

bool Dialated::operator<(Dialated& other){
	return value < other.value;
}

bool Dialated::operator<=(Dialated& other){
	return (to_int()) <= (other.to_int());
}

bool Dialated::operator>(Dialated& other){
	return to_int() > other.to_int();
}

bool Dialated::operator>=(Dialated& other){
	return to_int() >= other.to_int();
}


Dialated Dialated::operator/(Dialated& other){
	return value/other.value;
}

int Dialated::operator/(int val){
	return to_int()/val;
}

int Dialated::operator*(int val){
	return to_int()*val;
}

int Dialated::operator+(int val){
	return to_int() + val;
}

int Dialated::operator-(int val){
	return to_int() - val;
}

#endif
