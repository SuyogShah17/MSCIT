—------------Practical 1 Blockchain—-----------------------------------------------
import hashlib
import time
import json

# ==========================================
# PART 1: THE BLUEPRINTS (CLASS DEFINITIONS)
# ==========================================

class Block:
    def __init__(self, index, previous_hash, timestamp, data, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{json.dumps(self.data)}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4

    def create_genesis_block(self):
        return Block(0, "0", time.time(), "Genesis Block")

    def get_latest_block(self):
        return self.chain[-1]

    def mine_block(self, data):
        previous_block = self.get_latest_block()
        index = previous_block.index + 1
        timestamp = time.time()
        nonce = 0
        
        new_block = Block(index, previous_block.hash, timestamp, data, nonce)
        
        # The Mining Loop
        while not new_block.hash.startswith('0' * self.difficulty):
            new_block.nonce += 1
            new_block.hash = new_block.calculate_hash()
            
        self.chain.append(new_block)

    def dump_chain(self):
        print("\n--- Blockchain Ledger ---")
        for block in self.chain:
            print(f"Index: {block.index} | PrevHash: {block.previous_hash[:10]}... | Hash: {block.hash[:10]}... | Data: {block.data}")

# ==========================================
# PART 2: THE EXECUTION (RUNNING THE CODE)
# ==========================================

# 1. Turn on the factory
my_blockchain = Blockchain()

# 2. Feed it data
my_blockchain.mine_block({"sender": "Alice", "receiver": "Bob", "amount": 50})
my_blockchain.mine_block({"sender": "Bob", "receiver": "Charlie", "amount": 25})

# 3. Print the result
my_blockchain.dump_chain()



—------------Practical 2 (Practical 2A: "Write a Solidity program to create a smart contract that stores and retrieves a state variable."
Practical 2B: "Write a Solidity program to implement a counter that increments and decrements a value."
Practical 2C: "Write a Solidity program to create a basic calculator that performs addition, subtraction, multiplication, division, and modulo operations."
Practical 6: "Write a Solidity program to demonstrate the distinct behavior of the view and pure function modifiers."
)—----------
// SPDX-License-Identifier: MIT
pragma solidity >=0.7.0 <0.9.0;

contract UniversalCalc { 
    uint256 public num;

    function store(uint256 input) public {                              //2AStorage 
        num = input; 
    }



// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

contract UniversalCalc { 
    uint256 public num;                                            // 2A: Store a State Variable
    function store(uint256 input) public {                              
        num = input; 
    }

    function inc() public {                                    // 2B: Increment & Decrement Counter
        num++; 
    }
    function dec() public { 
        if (num >= 1) {
            num--; 
        }
    }

    function calc(uint256 a, uint256 b) public pure returns (            // 2C: Basic Calculator
        uint256 add, uint256 sub, uint256 mul, uint256 div, uint256 mod
    ) {
        add = a + b;
        sub = a >= b ? a - b : 0; 
        mul = a * b;
        if (b > 0) {
            div = a / b;
            mod = a % b;
        } else {
            div = 0;
            mod = 0;
        }
    }

    function retrieve() public view returns (uint256) {               // 6: View and Pure Functions
        return num;	                                                                //(Note: retrieve() also completes Prac 2A)
    }
    function display() public pure returns (string memory) {
        return "Practical performed by Suyog Shah Sathaye - 4";
    }
}



—------Practical 7// SPDX-License-Identifier: GPL-3.0—-----------------------------------------------
pragma solidity >=0.5.0 <0.9.0;

contract ClassAlloc {
    
    struct Student { string n; uint r; string c; }
    
    Student[30] public stds;
    uint public cnt; // Defaults to 0 automatically

    function display() public pure returns (string memory) {
        return "Practical performed by Suyog Shah Sathaye - 4";
    }

    function add(string memory _n, uint _r, string memory _c) public {
        require(cnt < 30, "Full");
        stds[cnt++] = Student(_n, _r, _c); // Saves the student AND adds 1 to cnt in one move
    }

    function alloc(uint r) public pure returns (string memory) {
        require(r > 0 && r <= 30, "Invalid");
        
        // The Waterfall Logic
        if (r <= 5) return "Class 45";
        if (r <= 10) return "Class 46";
        if (r <= 15) return "Class 47";
        if (r <= 20) return "Class 48";
        return "Class 49"; // Catches everything else (21-30)
    }
}

—--(Practical 3A: "Write a Solidity program to demonstrate Logical Operators: AND (&&), OR (||), and NOT (!)."
Practical 3B: "Write a Solidity program to demonstrate Assignment Operators: +=, -=, *=, and /=."
Practical 3C: "Write a Solidity program to demonstrate Bitwise Operators: AND (&), OR (|), XOR (^), NOT (~), Left Shift (<<), and Right Shift (>>)."
Practical 3D: "Write a Solidity program to demonstrate the Ternary Operator (condition ? true : false) for inline conditional logic."
Practical 3E: "Write a Solidity program to demonstrate Comparison Operators: equal (==), greater than or equal (>=), and less than or equal (<=)."
)--------------------------------
// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

contract Practical3_Operators {                               //3A: ASSIGNMENT OPERATORS\\
    uint256 public num;                                       // 3A: Storage Variable
    function setnum(uint256 _n) public {                      // 3A: Initialize Number
        num = _n; 
    }
    function addAssign(uint256 _n) public { num += _n; }      // 3A: Add and Assign (num = num + _n)
    function subAssign(uint256 _n) public { num -= _n; }      // 3A: Subtract and Assign
    function mulAssign(uint256 _n) public { num *= _n; }      // 3A: Multiply and Assign
    function divAssign(uint256 _n) public { num /= _n; }      // 3A: Divide and Assign

    uint256 public val1;                                   // 3B: BITWISE & TERNARY OPERATORS\\     
    uint256 public val2;                                      // 3B: Storage Variables
    
    function store(uint256 _v1) public { val1 = _v1; }        // 3B: Store Inputs
    function store1(uint256 _v2) public { val2 = _v2; }

    function bitwiseAnd() public view returns (uint256) { return val1 & val2; }   // 3B: Bitwise AND
    function bitwiseOr() public view returns (uint256) { return val1 | val2; }   // 3B: Bitwise OR
    function bitwiseXor() public view returns (uint256) { return val1 ^ val2; }  // 3B: Bitwise XOR
    function leftShift() public view returns (uint256) { return val1 << 1; }     // 3B: Left Shift
    function rightShift() public view returns (uint256) { return val1 >> 1; }    // 3B: Right Shift
    function Ternary() public view returns (string memory) {                     // 3B: Ternary Operator
        return (val1 > val2) ? "Val1 is Greater" : "Val2 is Greater/Equal";
    }

    uint256 public num1;                                     // 3C & 3D: COMPARISON OPERATORS\\
    uint256 public num2;                                      // 3C: Storage Variables
    
    function setnum1(uint256 _n1) public { num1 = _n1; }      // 3C: Store Inputs
    function setnum2(uint256 _n2) public { num2 = _n2; }

    // 'bool' means it returns True or False
    function isEqual() public view returns (bool) {           // 3C: Is Equal To
        return num1 == num2; 
    }
    function isGreaterOrEqual() public view returns (bool) { return num1 >= num2; } // 3D: Greater/Equal
    function isLessOrEqual() public view returns (bool) { return num1 <= num2; }    // 3D: Less/Equal

    function display() public pure returns (string memory) {                       
        return "Practical performed by Suyog Shah Sathaye - 4";
    }
}
—-----Practical 4: Function Overloading:  Mathematical Functions and Functions Overloading. Strategy: The easiest practical. You just write the same function name (sum or diff) but change the number of inputs.
—------------------
// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

contract Practical4_Overloading {                            // STATE VARIABLES (Blue View Buttons)\\
    uint256 public sumResult;                                 // Stores Addition
    uint256 public diffResult;                                // Stores Subtraction
    uint256 public proResult;                                 // Stores Multiplication
    uint256 public andResult;                                 // Stores Bitwise AND
    uint256 public orResult;                                  // Stores Bitwise OR

                                                        
    // 1. Arithmetical Overloading\\
    function sum(uint256 a, uint256 b) public { sumResult = a + b; }
    function sum(uint256 a, uint256 b, uint256 c) public { sumResult = a + b + c; }

    function diff(uint256 a, uint256 b) public { differenceResult = a - b; }
    function diff(uint256 a, uint256 b, uint256 c) public { differenceResult = a - b - c; }

    function pro(uint256 a, uint256 b) public { productResult = a * b; }
    function pro(uint256 a, uint256 b, uint256 c) public { productResult = a * b * c; }

    // 2. Bitwise Overloading\\
    function andRes(uint256 a, uint256 b) public { bitwiseAndResult = a & b; }
    function andRes(uint256 a, uint256 b, uint256 c) public { bitwiseAndResult = a & b & c; }

    function orRes(uint256 a, uint256 b) public { bitwiseOrResult = a | b; }
    function orRes(uint256 a, uint256 b, uint256 c) public { bitwiseOrResult = a | b | c; }

    // Required Display Function\\
    function display() public pure returns(string memory) { 
        return "Practical performed by Suyog Shah Sathaye - 4"; 
    }
}


—-----------prac 9 : Ternary & Type Casting  "To demonstrate inline conditional logic using the Ternary Operator, and to prevent an EVM Overflow crash by utilizing Type Casting (int256) to safely handle the massive integer values generated by the Bitwise NOT (~) operator." —-------------------------------
// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

contract TemperatureSensor { uint256 t1; uint256 t2;}                                           // 1. STATE VARIABLES (The Vault)

function recordTemps(uint256 _in1, uint256 _in2) public {              // 2. THE WRITE FUNCTION
t1 = _in1;
 	t2 = _in2;
 }

function getHottest() public view returns (uint256) {			
return t1 > t2 ? t1 : t2;                                          // Reads as: Is t1 greater than t2? If YES, return t1. If NO, return 12
   		 }

function calculateInvertedDifference() public view returns (int256) {           // 4. TYPE CASTING & BITWISE NOT PRACTICE
uint256 hottest = t1 > t2 ? t1 : t2;                                        // First, we find the hottest and coldest using Ternary logic
uint256 coldest = t1 > t2 ? t2 : t1;                         // The Defusal: We cast to int256 so the ~hottest doesn't crash the EVM
 return int256(coldest) * int256(~hottest);
 }
}

—------ prac 10: Write a solidity program to find modulus of Addition of two numbers with DD and modulus of multiplication of two numbers with MM where the two numbers are achieved by performing AND operation of DD and MM (from your Date of Birth) and OR operation of YY and YY (from your Year of Birth).—----------------------------
// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

contract Practical10 {                                       // STATE VARIABLES (Blue Buttons) \\
    uint256 public DD; 
    uint256 public MM;
    uint256 public YY1; 
    uint256 public YY2;
    uint256 public a1; 
    uint256 public a2;

    function storeDD(uint256 num) public { DD = num; }    		   // STORE DOB (Orange Buttons) \\
    function storeMM(uint256 num) public { MM = num; }
    function storeYY1(uint256 num) public { YY1 = num; }
    function storeYY2(uint256 num) public { YY2 = num; }

    function store1() public { a1 = DD & MM; }           		     // INTERMEDIATE BITWISE MATH \\
    function store2() public { a2 = YY1 | YY2; }           		  // (Calculates a1 and a2) \\

    function add_mod() public view returns (uint256) {      		 // FINAL OUTPUT (Blue Buttons) \\
        return addmod(a1, a2, DD);                                    		    // Formula: (a1 + a2) % DD
    }
    function mul_mod() public view returns (uint256) {  
        return mulmod(a1, a2, MM);                                           	       // Formula: (a1 * a2) % MM
    }

    function display() public pure returns(string memory) {        	        // IDENTITY OVERRIDE \\
        return "Practical performed by Suyog Shah Sathaye - 4";
    }
}









—----prac 5  Control Flow (Loops)  . Exact Aim: "To implement control flow in Solidity using for and while loops to iterate through array data, and to execute overflow-proof mathematical operations using the native addmod and mulmod functions."
—-------------
// SPDX-License-Identifier: MIT
pragma solidity >=0.7.0 <0.9.0;

contract Loops {
    uint256 public addmod1;
    uint256 public mulmod1;

    function calculate(uint256[] memory nums) public {
        // Updated to >= 6 so it won't crash if you add an extra number
        require(nums.length >= 6, "Need at least 6 numbers"); 
        
        uint256 n1; uint256 n2; uint256 n3;
        // FOR loop compressed
        for(uint256 i=0; i<3; i++) { if(i==0) n1=nums[i]; else if(i==1) n2=nums[i]; else n3=nums[i]; }
        addmod1 = addmod(n1, n2, n3); 

        uint256 n4; uint256 n5; uint256 n6; uint256 j=3;
        // WHILE loop compressed
        while(j<6) { if(j==3) n4=nums[j]; else if(j==4) n5=nums[j]; else n6=nums[j]; j++; }
        mulmod1 = mulmod(n4, n5, n6);
    }

    function display() public pure returns(string memory) { 
        return "Practical performed by Suyog Shah Sathaye - 4"; 
    }
}


—Prac 8:  Even/Odd, Prime, and Bitwise  8: Applied Algorithmic Logic.. Exact Aim: "To design a multi-step smart contract that integrates algorithmic loops (Prime and Even/Odd validation) with Bitwise math, demonstrating advanced state-tracking and conditional execution in the Ethereum Virtual Machine."
—---------------------
// SPDX-License-Identifier: MIT
pragma solidity >=0.7.0 <0.9.0;

contract Practical8 {
    uint256 num1; uint256 num2; 
    uint256 product; uint256 andRes; uint256 orRes;

    function store(uint256 num) public { num1 = num; }
    function store1(uint256 num) public { num2 = num; }

    function oddeven() public view returns (string memory) {
        if(num1 % 2 == 0) return "Is even"; else return "Is odd";
    }

    function prime() public view returns (string memory) {
        if (num2 <= 1) return "Not a prime";
        for (uint256 i = 2; i < num2; i++) { if (num2 % i == 0) return "Not a prime"; }
        return "Is prime";
    }

    function pro() public returns (uint256) { product = num1 * num2; return product; }
    function bitwiseAnd() public returns (uint256) { andRes = product & 2; return andRes; }
    function bitwiseOr() public returns (uint256) { orRes = product | 2; return orRes; }

    function finalop() public view returns (string memory) {
        if(andRes % 2 == 0 && orRes % 2 == 0) return "CONGRATULATIONS, YOU ARE SUCCESSFUL!!!";
        else return "CONGRATULATIONS, YOU ARE UNSUCCESSFUL!!!";
    }

    function display() public pure returns(string memory) { 
        return "Practical performed by Suyog Shah Sathaye - 4"; 
    }
}
































