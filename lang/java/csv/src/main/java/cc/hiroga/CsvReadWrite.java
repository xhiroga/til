import java.io.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.CSVPrinter;

public class CsvReadWrite{
	public static void main(String[] args){
		try{
			Reader reader = new FileReader("read.csv");
			FileWriter writer = new FileWriter("write.csv");

			Iterable<CSVRecord> records = CSVFormat.RFC4180.withHeader().parse(reader);
			CSVPrinter printer = new CSVPrinter(writer, CSVFormat.RFC4180);

			for(CSVRecord rc : records){
				printer.print(rc.get("Name"));
				printer.println();
			}
			reader.close();
			printer.flush();
			printer.close();

		}catch(IOException e){
			System.out.println(e);
		}
	}
}
