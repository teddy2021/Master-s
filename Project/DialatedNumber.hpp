// Kody Manastyrski
// CP 8316
// Computational Geometry
// Dialated Number hpp
//

#include<bitset>

using namespace std;

class Dialated{

		int value;
		bitset<64> mask;

	public:
		Dialated(int val);
		Dialated operator/(Dialated& other);
		Dialated operator+(const Dialated& other);
		Dialated operator-(const Dialated& other);
		Dialated operator-();
		int operator/(int val);
		int operator*(int val);
		int operator+(int val);
		int operator-(int val);
	
		bool operator==(Dialated& other);
		bool operator<(Dialated& other);
		bool operator<=(Dialated& other);
		bool operator>=(Dialated& other);
		bool operator>(Dialated& other);

	
		int to_int();


};
