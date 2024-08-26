package A1;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.StringTokenizer;

// Tuple class represents a mnemonic with its respective class, opcode, and length
class Tuple {
    String mnemonic, m_class, opcode;
    int length;

    Tuple() {}

    // Constructor to initialize a Tuple
    Tuple(String s1, String s2, String s3, String s4) {
        mnemonic = s1;
        m_class = s2;
        opcode = s3;
        length = Integer.parseInt(s4);
    }
}

// SymTuple class represents an entry in the Symbol Table with symbol, address, and length
class SymTuple {
    String symbol, address;
    int length;

    // Constructor to initialize a Symbol Tuple
    SymTuple(String s1, String s2, int i1) {
        symbol = s1;
        address = s2;
        length = i1;
    }
}

// LitTuple class represents an entry in the Literal Table with literal, address, and length
class LitTuple {
    String literal, address;
    int length;

    LitTuple() {}

    // Constructor to initialize a Literal Tuple
    LitTuple(String s1, String s2, int i1) {
        literal = s1;
        address = s2;
        length = i1;
    }
}

// Assembler_PassOne_V2 class handles the first pass of the assembler process
public class Assembler_PassOne_V2 {
    // Global Variables
    static int lc, iSymTabPtr = 0, iLitTabPtr = 0, iPoolTabPtr = 0;
    static int poolTable[] = new int[10];
    static Map<String, Tuple> MOT; // Mnemonic Opcode Table
    static Map<String, SymTuple> symtable; // Symbol Table
    static ArrayList<LitTuple> littable; // Literal Table
    static Map<String, String> regAddressTable; // Register Address Table
    static PrintWriter out_pass2, out_pass1, out_symtable, out_littable;
    static int line_no;

    public static void main(String[] args) throws Exception {
        initializeTables();
        System.out.println("===== PASS 1 OUTPUT =====\n");
        pass1();
    }

    // Method to perform the first pass of the assembler
    static void pass1() throws Exception {
        // Setting up input and output streams
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream("src/A1/input.txt")));
        out_pass1 = new PrintWriter(new FileWriter("src/A1/output_pass1.txt"), true);
        out_symtable = new PrintWriter(new FileWriter("src/A1/symtable.txt"), true);
        out_littable = new PrintWriter(new FileWriter("src/A1/littable.txt"), true);

        String s;
        lc = 0;
        while ((s = input.readLine()) != null) {
            StringTokenizer st = new StringTokenizer(s, " ", false);
            String s_arr[] = new String[st.countTokens()];
            for (int i = 0; i < s_arr.length; i++) {
                s_arr[i] = st.nextToken();
            }

            if (s_arr.length == 0) {
                continue; // Skip empty lines
            }

            int curIndex = 0;
            // If there's a label, add it to the Symbol Table
            if (s_arr.length == 3) {
                String label = s_arr[0];
                insertIntoSymTab(label, Integer.toString(lc));
                curIndex = 1;
            }

            // Process the current mnemonic
            String curToken = s_arr[curIndex];
            Tuple curTuple = MOT.get(curToken);

            String intermediateStr = "";
            // Handle different mnemonic classes (IS, AD, DL)
            if (curTuple.m_class.equalsIgnoreCase("IS")) {
                intermediateStr += lc + " (" + curTuple.m_class + "," + curTuple.opcode + ") ";
                lc += curTuple.length;
                intermediateStr += processOperands(s_arr[curIndex + 1]);
            } else if (curTuple.m_class.equalsIgnoreCase("AD")) {
                if (curTuple.mnemonic.equalsIgnoreCase("START")) {
                    intermediateStr += lc + " (" + curTuple.m_class + "," + curTuple.opcode + ") \n";
                    lc = Integer.parseInt(s_arr[curIndex + 1]);
                } else if (curTuple.mnemonic.equalsIgnoreCase("END")) {
                    intermediateStr += lc + " (" + curTuple.m_class + "," + curTuple.opcode + ") \n";
                    intermediateStr += processLTORG();
                } else if (curTuple.mnemonic.equalsIgnoreCase("LTORG")) {
                    intermediateStr += processLTORG();
                }
            } else if (curTuple.m_class.equalsIgnoreCase("DL")) {
                intermediateStr += lc + " (" + curTuple.m_class + "," + curTuple.opcode + ") ";
                if (curTuple.mnemonic.equalsIgnoreCase("DS")) {
                    lc += Integer.parseInt(s_arr[curIndex + 1]);
                } else if (curTuple.mnemonic.equalsIgnoreCase("DC")) {
                    lc += curTuple.length;
                }
                intermediateStr += "(C," + s_arr[curIndex + 1] + ") ";
            }

            // Print the instruction in the intermediate file
            out_pass1.println(intermediateStr);
        }

        // Close intermediate file
        out_pass1.flush();
        out_pass1.close();

        // Print symbol table
        System.out.println("===== Symbol Table =====");
        Iterator<SymTuple> it = symtable.values().iterator();
        while (it.hasNext()) {
            SymTuple tuple = it.next();
            String tableEntry = tuple.symbol + "\t" + tuple.address;
            out_symtable.println(tableEntry);
            System.out.println(tableEntry);
        }
        out_symtable.flush();
        out_symtable.close();

        // Print literal table
        System.out.println("===== Literal Table =====");
        for (int i = 0; i < littable.size(); i++) {
            LitTuple litTuple = littable.get(i);
            String tableEntry = litTuple.literal + "\t" + litTuple.address;
            out_littable.println(tableEntry);
            System.out.println(tableEntry);
        }
        out_littable.flush();
        out_littable.close();
    }

    // Process literals (LTORG) and assign addresses
    static String processLTORG() {
        LitTuple litTuple;
        String intermediateStr = "";
        for (int i = iPoolTable[iPoolTabPtr - 1]; i < littable.size(); i++) {
            litTuple = littable.get(i);
            litTuple.address = Integer.toString(lc++);
            intermediateStr += lc + " (DL,02) (C," + litTuple.literal + ") \n";
        }
        // Add new entry to pool table
        poolTable[iPoolTabPtr] = iLitTabPtr;
        iPoolTabPtr++;
        return intermediateStr;
    }

    // Process the operands in an instruction
    static String processOperands(String operands) {
        StringTokenizer st = new StringTokenizer(operands, ",", false);
        String s_arr[] = new String[st.countTokens()];
        for (int i = 0; i < s_arr.length; i++) {
            s_arr[i] = st.nextToken();
        }

        String intermediateStr = "";
        for (int i = 0; i < s_arr.length; i++) {
            String curToken = s_arr[i];
            if (curToken.startsWith("=")) {
                // Operand is a literal
                StringTokenizer str = new StringTokenizer(curToken, "=", false);
                String tokens[] = new String[str.countTokens()];
                for (int j = 0; j < tokens.length; j++) {
                    tokens[j] = str.nextToken();
                }
                String literal = tokens[1];
                insertIntoLitTab(literal, "");
                intermediateStr += "(L," + (iLitTabPtr - 1) + ") ";
            } else if (regAddressTable.containsKey(curToken)) {
                // Operand is a register name
                intermediateStr += "(RG," + regAddressTable.get(curToken) + ") ";
            } else {
                // Operand is a symbol
                insertIntoSymTab(curToken, "");
                intermediateStr += "(S," + (iSymTabPtr - 1) + ") ";
            }
        }
        return intermediateStr;
    }

    // Insert a symbol into the Symbol Table
    static void insertIntoSymTab(String symbol, String address) {
        // Check if the symbol is already present
        if (symtable.containsKey(symbol)) {
            // Update the address if symbol exists
            SymTuple s = symtable.get(symbol);
            s.address = address;
        } else {
            // Add a new symbol entry
            symtable.put(symbol, new SymTuple(symbol, address, 1));
            iSymTabPtr++;
        }
    }

    // Insert a literal into the Literal Table
    static void insertIntoLitTab(String literal, String address) {
        // Add a new literal entry
        littable.add(iLitTabPtr, new LitTuple(literal, address, 1));
        iLitTabPtr++;
    }

    // Initialize tables for mnemonics, symbols, literals, and registers
    static void initializeTables() throws Exception {
        symtable = new LinkedHashMap<>();
        littable = new ArrayList<>();
        regAddressTable = new HashMap<>();
        MOT = new HashMap<>();

        // Load Mnemonic Opcode Table (MOT) from file
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("src/A1/mot.txt")));
        String s, mnemonic;
        while ((s = br.readLine()) != null) {
            StringTokenizer st = new StringTokenizer(s, " ", false);
            mnemonic = st.nextToken();
            MOT.put(mnemonic, new Tuple(mnemonic, st.nextToken(), st.nextToken(), st.nextToken()));
        }
        br.close();

        // Initialize register address table
        regAddressTable.put("AREG", "1");
        regAddressTable.put("BREG", "2");
        regAddressTable.put("CREG", "3");
        regAddressTable.put("DREG", "4");

        // Initialize pool table
        poolTable[iPoolTabPtr] = iLitTabPtr;
        iPoolTabPtr++;
    }
}
