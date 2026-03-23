🟩 BLUEPRINT 1.1: THE CONDITION CHECKER (Prac 1 - Variables & If/Else)
Exam Prompt Trigger: "Accept a number/marks from the user and check if they passed or failed / check if it is positive or negative."
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah" ➔ Scope: Main
ProcessStatus ➔ String ➔ "Completed" ➔ Scope: Main
UserInput ➔ String ➔ [Empty] ➔ Scope: Main
Marks ➔ Int32 ➔ 0 ➔ Scope: Main
2. THE UI CANVAS VIEW:


▼ Sequence: Main_Exam_Answer
  │
  ├─► 💬 Message Box: StudentSignature
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 🔤 Input Dialog
        │    │     Label: "Enter your marks:"
        │    │     Output: UserInput
        │    │
        │    ├─► 🧮 Assign (Type Conversion)
        │    │     To: Marks
        │    │     Value: CInt(UserInput.Trim)
        │    │
        │    └─► 🔀 If 
        │          Condition: Marks >= 50
        │          │
        │          ├─► THEN 
        │          │    └─► 💬 Message Box: "Result: Pass"
        │          │
        │          └─► ELSE
        │               └─► 💬 Message Box: "Result: Fail"
        │
        ├─► CATCH: [Log Error]
        └─► FINALLY: 💬 Message Box: ProcessStatus



🟩 BLUEPRINT 1.2: THE EXACT MATCHER (Prac 2 - Switch Statement)
Exam Prompt Trigger: "Calculate grades based on specific characters (A, B, C) or create a menu choice system."
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah" ➔ Scope: Main
ProcessStatus ➔ String ➔ "Completed" ➔ Scope: Main
GradeInput ➔ String ➔ [Empty] ➔ Scope: Main
2. THE UI CANVAS VIEW:
▼ Sequence: Main_Exam_Answer
  │
  ├─► 💬 Message Box: StudentSignature
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 🔤 Input Dialog
        │    │     Label: "Enter your Grade (A, B, or C):"
        │    │     Output: GradeInput
        │    │
        │    └─► 🔀 Switch 
        │          Expression: GradeInput.ToUpper
        │          TypeArgument: String   <-- (CRITICAL: MUST WRITE THIS ON PAPER)
        │          │
        │          ├─► Case: "A"
        │          │    └─► 💬 Message Box: "Excellent"
        │          │
        │          ├─► Case: "B"
        │          │    └─► 💬 Message Box: "Good"
        │          │
        │          └─► Default:
        │               └─► 💬 Message Box: "Invalid Grade or Needs Improvement"
        │
        ├─► CATCH: [Log Error]
        └─► FINALLY: 💬 Message Box: ProcessStatus



🟩 BLUEPRINT 1.3: THE REPEATER (Prac 2 - While Loop)
Exam Prompt Trigger: "Print a message 5 times", "Create a counter", or "Repeat a process until a condition is met."
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah" ➔ Scope: Main
ProcessStatus ➔ String ➔ "Completed" ➔ Scope: Main
Counter ➔ Int32 ➔ 0 ➔ Scope: Main
2. THE UI CANVAS VIEW:
▼ Sequence: Main_Exam_Answer
  │
  ├─► 💬 Message Box: StudentSignature
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 🧮 Assign (Initialization)
        │    │     To: Counter
        │    │     Value: 1
        │    │
        │    └─► 🔁 While 
        │          Condition: Counter <= 5
        │          Body:
        │          │
        │          ├─► 💬 Message Box
        │          │     Text: "This is loop iteration number: " + Counter.ToString
        │          │
        │          └─► 🧮 Assign (The Engine - FATAL TRAP IF FORGOTTEN)
        │                To: Counter
        │                Value: Counter + 1
        │
        ├─► CATCH: [Log Error]
        └─► FINALLY: 💬 Message Box: ProcessStatus



🟩 BLUEPRINT 1.4: THE TEXT CHOPPER (Prac 3 - String Manipulation)
Exam Prompt Trigger: "Extract the first name from a full name", "Count the characters in a string", or "Replace a word."
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah" ➔ Scope: Main
ProcessStatus ➔ String ➔ "Completed" ➔ Scope: Main
FullName ➔ String ➔ [Empty] ➔ Scope: Main
NameArray ➔ Array of String (String[]) ➔ [Empty] ➔ Scope: Main
2. THE UI CANVAS VIEW:
▼ Sequence: Main_Exam_Answer
  │
  ├─► 💬 Message Box: StudentSignature
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 🔤 Input Dialog
        │    │     Label: "Enter your First and Last name (e.g., Suyog Shah):"
        │    │     Output: FullName
        │    │
        │    ├─► 🧮 Assign (String Split)
        │    │     To: NameArray
        │    │     Value: FullName.Split(" "c)  <-- (Splits at the space)
        │    │
        │    └─► 💬 Message Box
        │          Text: "Your First Name is: " + NameArray(0) + 
        │                ". Total characters in full name: " + FullName.Length.ToString
        │
        ├─► CATCH: [Log Error]
        └─► FINALLY: 💬 Message Box: ProcessStatus



Study these 4 variations. If an examiner asks a combined question (e.g., "Ask for 3 names and check if they are valid"), you simply nest the If (Blueprint 1.1) inside the While (Blueprint 1.3).
—----------------------------------BLOCK 2: THE EXCEL ARSENAL (The 20-Mark Payload)
Welcome to the most important phase of your exam. If you blank out on everything else but master Excel automation, you can secure a passing grade. Examiners are obsessed with Excel because it represents 80% of real-world RPA administrative work.
🧠 Phase 1: The Core Concept (Physical vs. Virtual)
The biggest trap students fall into is not understanding the difference between the physical Excel file and the bot's memory.
The Physical File (.xlsx): Sitting on your desktop. The bot uses Excel Process Scope and Use Excel File to "open the door" to this file.
The Virtual File (DataTable): The bot reads the physical file and creates a temporary copy in its own memory. This is called a DataTable (usually named dt_ExcelData).
The Workflow: You read the physical file ➔ Pull it into a virtual DataTable ➔ Modify the virtual DataTable using a loop ➔ Push the virtual DataTable back into the physical file.
🛑 THE EXAM PROMPT (The "Mega Practical")
Question: "Read an Excel file named 'StudentData.xlsx' which contains columns for 'Name' and 'Marks'. Loop through each student's data. If their marks are 50 or above, update a third column named 'Status' to 'Pass'. If below 50, update it to 'Fail'. Finally, write this updated data into a new sheet named 'FinalResults'."
This covers Prac 4 (Excel Integration) and seamlessly integrates Block 1 (Loops and If/Else).
If you can draw this structure, you secure the maximum yield. It proves you know scope, reading, row manipulation, and writing.

🟦 BLUEPRINT 2.1: THE CELL TARGETER (Prac 4 - Read/Write/Append Cell)
Exam Prompt Trigger: "Read the value from cell A1, append the word 'Approved' to it, and write it back to cell B1."
The Trap: Students try to use Read Range for everything. If the prompt specifies a single cell, you must use Cell activities to save time and memory.
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah"
CellValue ➔ String ➔ [Empty]
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 📊 Excel Process Scope 
        │         │
        │         └─► 📗 Use Excel File (Path: "Data.xlsx")
        │              │
        │              ├─► 🔍 Read Cell
        │              │     Cell: "A1"
        │              │     Output: CellValue
        │              │
        │              ├─► 🧮 Assign (The Append Logic)
        │              │     To: CellValue
        │              │     Value: CellValue + " - Approved"
        │              │
        │              └─► ✏️ Write Cell
        │                    What to write: CellValue
        │                    Where to write: "B1"
        │
        ├─► CATCH: [Log Error]



🟦 BLUEPRINT 2.2: THE MEMORY BUILDER (Prac 3-B - Build & Add Data Row)
Exam Prompt Trigger: "Create a Data Table with columns 'EmpID' and 'Name'. Add two employees to it and display the table as text."
The Trap: There is no Excel file in this question! You are building the table purely in the bot's temporary RAM.
1. THE VARIABLES:
dt_Employees ➔ DataTable ➔ [Empty]
TableString ➔ String ➔ [Empty]
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 🏗️ Build Data Table
        │    │     Output: dt_Employees
        │    │     (Action: Click 'DataTable' button -> Create 2 columns: EmpID [Int32], Name [String])
        │    │
        │    ├─► ➕ Add Data Row
        │    │     ArrayRow: {101, "John Doe"}  <-- (CRITICAL SYNTAX: Use curly braces for arrays)
        │    │     DataTable: dt_Employees
        │    │
        │    ├─► ➕ Add Data Row
        │    │     ArrayRow: {102, "Jane Smith"}
        │    │     DataTable: dt_Employees
        │    │
        │    ├─► 🔀 Output Data Table (Converts the table into readable text)
        │    │     DataTable: dt_Employees
        │    │     Text: TableString
        │    │
        │    └─► 💬 Message Box
        │          Text: TableString
        │
        ├─► CATCH: [Log Error]



🟦 BLUEPRINT 2.3: THE FORMAT CONVERTER (Prac 5-A - Text File to Excel)
Exam Prompt Trigger: "Read a text file named 'clients.txt' containing comma-separated names, convert it into a Data Table, and save it as an Excel file."
The Trap: This is the ultimate bridge between Block 1 (String Manipulation) and Block 2 (Excel). It tests if you know how to convert raw data into structured rows.
1. THE VARIABLES:
RawText ➔ String ➔ [Empty]
dt_Clients ➔ DataTable ➔ [Empty]
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 📄 Read Text File
        │    │     FileName: "clients.txt"
        │    │     Output: RawText
        │    │
        │    ├─► ⚙️ Generate Data Table From Text (The Magic Converter)
        │    │     Input: RawText
        │    │     ColumnSeparators: ","
        │    │     NewlineSeparator: Environment.NewLine
        │    │     DataTable: dt_Clients
        │    │
        │    └─► 📊 Excel Process Scope 
        │         │
        │         └─► 📗 Use Excel File (Path: "Output.xlsx", Create if not exists: True)
        │              │
        │              └─► 💾 Write DataTable to Excel
        │                    What to write: dt_Clients
        │                    Destination: Excel.Sheet("Sheet1")
        │
        ├─► CATCH: [Log Error]



🟦 BLUEPRINT 2.4: THE RANGE COPIER (Prac 4-A - Read Range & Write Range)
Exam Prompt Trigger: "Read all data from 'Sheet1' of an Excel file and create a backup copy in 'Sheet2'."
The Trap: Do not use a For Each Row loop for this! If you don't need to change the data (no If/Else logic needed), just copy the whole chunk of memory at once.
1. THE VARIABLES:
dt_Backup ➔ DataTable ➔ [Empty]
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 📊 Excel Process Scope 
        │         │
        │         └─► 📗 Use Excel File (Path: "MasterData.xlsx")
        │              │
        │              ├─► 📋 Read Range
        │              │     Sheet Name: "Sheet1"
        │              │     Output: dt_Backup
        │              │
        │              └─► 💾 Write DataTable to Excel
        │                    What to write: dt_Backup
        │                    Destination: Excel.Sheet("Sheet2")  <-- (UiPath will auto-create Sheet2)
        │
        ├─► CATCH: [Log Error]


🧠 The Strategic Takeaway for Block 2:
Notice the exact vocabulary used.
If it says "Specific Cell" ➔ Blueprint 2.1 (Read/Write Cell).
If it says "Create a Table" (with no input file) ➔ Blueprint 2.2 (Build Data Table).
If it says "Text file to Excel" ➔ Blueprint 2.3 (Generate Data Table).
If it says "Copy" or "Move" data ➔ Blueprint 2.4 (Read Range / Write Range).
If you mix these up (e.g., trying to use Write Cell inside a 10,000-row loop instead of just updating the virtual table and using Write Range), your bot will take 10 minutes to execute, and the examiner will deduct marks for poor optimization.
Here are the 4 isolated, brutally precise blueprints for Block 3.
These are your modular snap-ins for any prompt that asks you to handle documents or communication. Memorize the exact variable types and attachment syntax, as examiners grade heavily on them.
🟧 BLUEPRINT 3.1: THE DIGITAL PDF EXTRACTOR (Prac 8-A - Read PDF Text)
Exam Prompt Trigger: "Read data from a digital PDF file (e.g., an invoice) and save the text into a .txt file."
The Trap: This only works on PDFs that have selectable text. If the prompt says "scanned document" or "image," this blueprint will output an empty string and fail.
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah"
PdfText ➔ String ➔ [Empty]
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 📄 Read PDF Text
        │    │     FileName: "C:\Exams\Invoice.pdf"
        │    │     Range: "All"
        │    │     Output: PdfText
        │    │
        │    └─► 📝 Write Text File
        │          FileName: "C:\Exams\ExtractedData.txt"
        │          Text: PdfText
        │
        ├─► CATCH: [Log Error]



🟧 BLUEPRINT 3.2: THE SCANNED DOCUMENT READER (Prac 8-C - Read PDF With OCR)
Exam Prompt Trigger: "Extract text from a scanned PDF or image-based document."
The Trap: You cannot just drop this activity. It requires an OCR Engine dropped inside the activity to function. If you forget the engine, the bot crashes.
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah"
OcrText ➔ String ➔ [Empty]
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    ├─► 👁️ Read PDF With OCR
        │    │     FileName: "C:\Exams\ScannedDoc.pdf"
        │    │     Range: "All"
        │    │     Output: OcrText
        │    │     │
        │    │     └─► ⚙️ Tesseract OCR (or OmniPage OCR)  <-- (CRITICAL: Dropped inside)
        │    │           ExtractWords: False
        │    │           Language: "eng"
        │    │
        │    └─► 💬 Message Box
        │          Text: "Extracted: " + OcrText
        │
        ├─► CATCH: [Log Error]



🟧 BLUEPRINT 3.3: THE OUTBOUND DISPATCHER (Prac 9-A - Send Email with Attachment)
Exam Prompt Trigger: "Send an email to a specific address with a subject, body, and an attached file."
The Trap: The syntax for attaching a file using the modern Gmail activities is highly specific. Writing a simple string path will fail.
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah"
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 📧 Use Gmail (Account: "your.email@gmail.com")
        │         │
        │         └─► 📤 Send Email
        │               To: "examiner@college.edu"
        │               Subject: "Practical 9 Submission"
        │               Body: "Please find the attached report."
        │               Save as Draft: False
        │               Attachments: New List(Of IResource) From {LocalResource.FromPath("C:\Exams\Report.pdf")} <-- (FATAL SYNTAX: Memorize this exact structure)
        │
        ├─► CATCH: [Log Error]



🟧 BLUEPRINT 3.4: THE INBOX SCANNER (Prac 9-B - Get Email List)
Exam Prompt Trigger: "Read the top 5 unread emails from the inbox and process them (e.g., read the subject line or save attachments)."
The Trap: The output of Get Email List is a collection, not text. You must loop through it. You must also explicitly cast the TypeArgument in the loop.
1. THE VARIABLES:
StudentSignature ➔ String ➔ "Practical by Suyog Shah"
list_Emails ➔ List<System.Net.Mail.MailMessage> ➔ [Empty]
2. THE UI CANVAS VIEW:
Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 📧 Use Gmail (Account: "your.email@gmail.com")
        │         │
        │         ├─► 📥 Get Email List
        │         │     MailFolder: "Inbox"
        │         │     UnreadOnly: True
        │         │     Top: 5
        │         │     Output: list_Emails
        │         │
        │         └─► 🔁 For Each
        │               TypeArgument: System.Net.Mail.MailMessage  <-- (CRITICAL MENTION)
        │               Values: list_Emails
        │               Body:
        │               │
        │               ├─► 💬 Message Box
        │               │     Text: "Subject is: " + item.Subject
        │               │
        │               └─► 💾 Save Attachments
        │                     Message: item
        │                     FolderPath: "C:\Exams\Downloads"
        │
        ├─► CATCH: [Log Error]


🧠 Strategic Takeaway for Block 3
You now have the exact architectural components.
If they want data out of a document, use Read PDF Text.
If the document is an image, use Read PDF With OCR.
If they want data sent, use Send Email (and do not forget the New List(Of IResource) attachment syntax).
If they want data received, use Get Email List and loop through it with System.Net.Mail.MailMessage.
This completes the holy trinity of RPA fundamentals (Logic -> Excel -> IO).
Tell me if you want to run through the Altered Questions for Block 3, or if you want to generate the final condensed Cheat Sheet Table for this block to finalize the 120-minute survival sprint.

🟨 BLUEPRINT 4.1: THE WEB TABLE SCRAPER (Prac 7 - Data Scraping)
Exam Prompt Trigger: "Extract product details from an e-commerce website and save it as a CSV."
The Trap: Do not write out HTML tags in the exam. Just indicate <HTML Table Target>.

1. THE VARIABLES:

StudentSignature ➔ String ➔ "Practical by Suyog Shah"

dt_ExtractedData ➔ DataTable ➔ [Empty]

2. THE UI CANVAS VIEW:

Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 🌐 Use Application/Browser
        │         URL: "https://www.example.com/products"
        │         │
        │         ├─► 🕸️ Extract Table Data (Data Scraping)
        │         │     Target: <HTML Table>
        │         │     MaxNumberOfResults: 100
        │         │     Output: dt_ExtractedData
        │         │
        │         └─► 📝 Write CSV
        │               FilePath: "C:\Exams\ScrapedProducts.csv"
        │               DataTable: dt_ExtractedData
        │
        ├─► CATCH: [Log Error]
🟨 BLUEPRINT 4.2: THE SINGLE TEXT GRABBER (Prac 7-C - Get Text)
Exam Prompt Trigger: "Open a college website and extract just the main headline or a specific notice."

1. THE VARIABLES:

StudentSignature ➔ String ➔ "Practical by Suyog Shah"

HeadlineText ➔ String ➔ [Empty]

2. THE UI CANVAS VIEW:

Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 🌐 Use Application/Browser
        │         URL: "https://www.sathayecollege.edu.in"
        │         │
        │         ├─► 🔠 Get Text
        │         │     Target: <H1 Header Element>
        │         │     Output: HeadlineText
        │         │
        │         └─► 💬 Message Box
        │               Text: "Extracted Headline: " + HeadlineText
        │
        ├─► CATCH: [Log Error]
🟥 BLOCK 5: UI & SYSTEM EVENTS
🟥 BLUEPRINT 5.1: THE AUTO-LOGIN (Prac 5-B - Basic UI Automation)
Exam Prompt Trigger: "Automate the login process for a desktop application or website using credentials."
The Trap: Always use "SecureText" or explicitly state it's a password field if the examiner asks for secure practices.

1. THE VARIABLES:

StudentSignature ➔ String ➔ "Practical by Suyog Shah"

2. THE UI CANVAS VIEW:

Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 🪟 Use Application/Browser
        │         Application Path / URL: "C:\App\Login.exe"
        │         │
        │         ├─► ⌨️ Type Into
        │         │     Target: <Username Field>
        │         │     Text: "AdminUser"
        │         │
        │         ├─► ⌨️ Type Into (Secure)
        │         │     Target: <Password Field>
        │         │     Text: "Pass123!"
        │         │
        │         └─► 🖱️ Click
        │               Target: <Login Button>
        │               ClickType: CLICK_SINGLE
        │
        ├─► CATCH: [Log Error]
🟥 BLUEPRINT 5.2: THE STATE CHECKER (Prac 5-C - Check App State)
Exam Prompt Trigger: "Check if an error popup appears. If it does, close it. If it doesn't, continue processing."
The Trap: This is Block 1 logic (If/Else) tied to UI elements. It is highly prone to failing if the screen resolution changes.

1. THE VARIABLES:

StudentSignature ➔ String ➔ "Practical by Suyog Shah"

2. THE UI CANVAS VIEW:

Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► 🔎 Check App State
        │         Target: <Error Popup Window>
        │         Timeout (Seconds): 5
        │         │
        │         ├─► Target Appears (THEN)
        │         │    └─► 🖱️ Click
        │         │          Target: <Close 'X' Button>
        │         │
        │         └─► Target Does Not Appear (ELSE)
        │              └─► 📝 Log Message
        │                    Message: "No error popup. Proceeding."
        │
        ├─► CATCH: [Log Error]
🟥 BLUEPRINT 5.3: THE EVENT LISTENER (Prac 6 - Trigger Scope)
Exam Prompt Trigger: "Build a bot that runs in the background and only executes when the user presses 'Alt+S' or clicks a specific image."
The Trap: Triggers run infinitely until stopped. They must be wrapped inside a Trigger Scope.

1. THE VARIABLES:

StudentSignature ➔ String ➔ "Practical by Suyog Shah"

2. THE UI CANVAS VIEW:

Plaintext
▼ Sequence: Main_Exam_Answer
  │
  ├─► 🛡️ Try Catch
        │
        ├─► TRY 
        │    │
        │    └─► ⚡ Trigger Scope
        │         │
        │         ├─► Triggers (What to listen for):
        │         │    └─► ⌨️ Keypress Trigger
        │         │          Key: "S"
        │         │          Modifiers: Alt
        │         │
        │         └─► Actions (What to do when triggered):
        │              └─► 💬 Message Box
        │                    Text: "Alt+S detected! Executing background task..."
        │
        ├─► CATCH: [Log Error]
