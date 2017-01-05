package com.anurag.pr2_ir;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

public class App {

	String indexFilePath;

	App(String indexFile) {
		indexFilePath = indexFile;
	}

	/*
	 * This function is used to parse the index given as input to this program
	 */
	public HashMap<String, LinkedList<Integer>> parse() throws Exception {
		// File fil_path = new File("/home/andi/index");

		File filPath = new File(indexFilePath);

		Directory index = FSDirectory.open(filPath.toPath());
		IndexReader reader = DirectoryReader.open(index);

		String textArr[] = { "text_nl", "text_fr", "text_de", "text_ja", "text_ru", "text_pt", "text_es", "text_it",
				"text_da", "text_no", "text_sv" };
		int numTextField = 11;

		HashMap<String, LinkedList<Integer>> map = new HashMap<String, LinkedList<Integer>>();
		for (int i = 0; i < numTextField; i++) {

			Terms term = MultiFields.getTerms(reader, textArr[i]);
			TermsEnum tpos = term.iterator();

			while (tpos.next() != null) {
				BytesRef text = tpos.term();
				String termString = text.utf8ToString();

				LinkedList<Integer> list = new LinkedList<Integer>();
				PostingsEnum postings = MultiFields.getTermPositionsEnum(reader, textArr[i], text);
				while ((postings.nextDoc()) != PostingsEnum.NO_MORE_DOCS) {
					list.add(postings.docID());

				}

				map.put(termString, list);

			}
		}
		reader.close();
		return map;

	}

	public void taatAND(HashMap<String, LinkedList<Integer>> map, String qterm, File f) throws Exception {
		String[] q = qterm.split(" ");
		LinkedList<LinkedList<Integer>> lists = new LinkedList<LinkedList<Integer>>();
		String doc = "";
		/*
		 * Splitting query terms and constructing the linked list from term
		 * posting lists of the terms
		 */
		for (int i = 0; i < q.length; i++) {
			FileUtils.write(f, "GetPostings\n", StandardCharsets.UTF_8, true);
			doc = q[i] + "\n";
			FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
			LinkedList<Integer> temp = new LinkedList<Integer>();
			temp.addAll(map.get(q[i]));
			lists.add(temp);
			doc = "Postings list: ";
			for (int j = 0; j < temp.size()-1; j++) {

				doc = doc + temp.get(j) + " ";
			}
			doc = doc + temp.getLast();
			doc += "\n";
			FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		}
		int[] pointers = new int[lists.size()];
		for (int i = 0; i < pointers.length; i++)
			pointers[i] = 0;
		int comparisons = 0;
		LinkedList<Integer> res = new LinkedList<Integer>();
		LinkedList<Integer> temp = new LinkedList<Integer>();
		temp.addAll(lists.get(0));

		for (int i = 1; i < lists.size(); i++) {
			/* Resetting the result linked list */
			res.clear();

			LinkedList<Integer> l1 = new LinkedList<Integer>();

			l1 = lists.get(i);
			pointers[i - 1] = 0;

			while (pointers[i] < l1.size() && pointers[i - 1] < temp.size()) {
				/* To check if the docId matches for two terms */
				if ((l1.get(pointers[i]).equals(temp.get(pointers[i - 1])))) {
					/*
					 * If matched then increment the comparison count and add
					 * the result
					 */
					comparisons++;
					res.add(l1.get(pointers[i]));

					pointers[i] = pointers[i] + 1;
					pointers[i - 1] = pointers[i - 1] + 1;

				} else if (l1.get(pointers[i]) < temp
						.get(pointers[i - 1])) /*
												 * Else determine if less than
												 * and increment the count
												 */
				{
					comparisons++;
					pointers[i] = pointers[i] + 1;
				} else if (l1.get(pointers[i]) > temp.get(pointers[i - 1])) {
					comparisons++;
					pointers[i - 1] = pointers[i - 1] + 1;
				}
			}

			temp.clear();
			temp.addAll(res);

		}
		FileUtils.write(f, "TaatAnd\n", StandardCharsets.UTF_8, true);
		FileUtils.write(f, qterm + "\n", StandardCharsets.UTF_8, true);
		doc = "";
		doc += "Results: ";
		res.clear();
		res.addAll(temp);
		temp.clear();
		if (res.size() > 0) {
			for (int j = 0; j < res.size()-1; j++) {
				doc = doc + res.get(j) + " ";
			}
			doc = doc + res.getLast();
		} else
			doc = doc + "empty";

		doc += "\n";
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of documents in results: %d\n", res.size());
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of comparisons: %d\n", comparisons);
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		res.clear();
		lists.clear();

	}

	public void taatOR(HashMap<String, LinkedList<Integer>> map, String qterm, File f) throws Exception {
		String[] q = qterm.split(" ");
		LinkedList<LinkedList<Integer>> lists = new LinkedList<LinkedList<Integer>>();
		String doc = "";

		/* Extract the posting lists for the terms */
		for (int i = 0; i < q.length; i++) {
			lists.add(map.get(q[i]));
		}
		int[] pointers = new int[lists.size()];
		for (int i = 0; i < pointers.length; i++)
			pointers[i] = 0;
		int comparisons = 0;
		LinkedList<Integer> res = new LinkedList<Integer>();
		LinkedList<Integer> temp = new LinkedList<Integer>();
		/* Add initial linked list into the intermediate result */
		temp.addAll(lists.get(0));
		LinkedList<Integer> l1 = new LinkedList<Integer>();
		/*
		 * For taatOr logic we need to accumulate each term by comparing so
		 * check for comparisons and then add
		 */
		for (int i = 1; i < lists.size(); i++) {
			/* Iterate through each linked list for each cycle */
			int idx1 = 0, idx2 = 0;
			res.clear();
			l1 = lists.get(i);
			pointers[i - 1] = 0;

			/* Repeat until both list isn't exhausted */
			while (pointers[i] < l1.size() && pointers[i - 1] < temp.size()) {

				if ((l1.get(pointers[i]).equals(temp.get(pointers[i - 1])))) {

					comparisons++;
					res.add(l1.get(pointers[i]));
					pointers[i] = pointers[i] + 1;
					pointers[i - 1] = pointers[i - 1] + 1;

				} else if (l1.get(pointers[i]) < temp.get(pointers[i - 1])) {
					comparisons++;
					res.add(l1.get(pointers[i]));
					pointers[i] = pointers[i] + 1;
				} else if (l1.get(pointers[i]) > temp.get(pointers[i - 1])) {
					comparisons++;
					res.add(temp.get(pointers[i - 1]));
					pointers[i - 1] = pointers[i - 1] + 1;
				}
				idx1 = i;
				idx2 = i - 1;
			}

			i = idx1;
			int k = idx2;
			if (pointers[i] < l1.size()) {
				for (int l = pointers[i]; l < l1.size(); l++) {
					res.add(l1.get(l));
				}
			}

			if (pointers[k] < temp.size()) {

				for (int l = pointers[k]; l < temp.size(); l++) {
					res.add(temp.get(l));
				}
			}
			temp.clear();
			temp.addAll(res);

		}
		FileUtils.write(f, "TaatOr\n", StandardCharsets.UTF_8, true);
		FileUtils.write(f, qterm + "\n", StandardCharsets.UTF_8, true);
		doc = "Results: ";
		res.clear();
		res.addAll(temp);
		for (int j = 0; j < res.size()-1; j++) {
			doc = doc + res.get(j) + " ";
		}
		doc = doc + res.getLast();
		doc += "\n";
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of documents in results: %d\n", res.size());
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of comparisons: %d\n", comparisons);
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);

	}

	/*
	 * Check if the docId matches at the current positions of the linked lists
	 */

	/*
	 * Another optimization applied here to reduce the number of comparisons.
	 * Instead of computing minValues individually for AND operation incremented
	 * all the docIds which are not maxVal among those being compared which
	 * resulted in reduction of number of comparisons by 20% than the
	 * conventional method
	 */
	public LinkedList<Integer> checkIfMatched(LinkedList<LinkedList<Integer>> lists, int[] pointers) {

		boolean flag = true;
		int docId = lists.get(0).get(pointers[0]);
		for (int i = 1; i < lists.size(); i++) {

			if (docId != lists.get(i).get(pointers[i])) {
				flag = false;

			}
		}
		LinkedList<Integer> minPos = new LinkedList<Integer>();
		int maxVal = docId;
		if (!flag) {
			for (int i = 0; i < lists.size(); i++) {
				int currdocId = lists.get(i).get(pointers[i]);
				if (currdocId > maxVal) {
					maxVal = currdocId;
				}
			}

			for (int i = 0; i < lists.size(); i++) {
				int currdocId = lists.get(i).get(pointers[i]);
				if (currdocId != maxVal)
					minPos.add(i);

			}
		}
		/*
		 * Last value is always a flag to determine if the docIds matched or not
		 */
		if (flag)
			minPos.add(1);
		else
			minPos.add(0);
		return minPos;
	}

	public boolean incrementAll(int[] pointers, LinkedList<LinkedList<Integer>> lists) {
		boolean ret_val = false;

		for (int i = 0; i < pointers.length; i++) {
			if (pointers[i] == lists.get(i).size() - 1) {
				ret_val = true;
				break;
			} else
				pointers[i] = pointers[i] + 1;
		}
		return ret_val;
	}

	public void daatAND(HashMap<String, LinkedList<Integer>> map, String qterm, File f) throws Exception {
		String[] q = qterm.split(" ");
		LinkedList<LinkedList<Integer>> lists = new LinkedList<LinkedList<Integer>>();
		// Construct the linked list of posting lists
		for (int i = 0; i < q.length; i++) {
			lists.add(map.get(q[i]));

		}
		int[] pointers = new int[lists.size()];
		for (int i = 0; i < pointers.length; i++)
			pointers[i] = 0;
		int comparisons = 0;
		LinkedList<Integer> res = new LinkedList<Integer>();
		res.clear();
		boolean listExhausted = false;
		// String out;
		while (listExhausted != true) {
			for (int i = 0; i < lists.size(); i++) {
				if (listExhausted == true) {
					break;
				}

				LinkedList<Integer> minPos = new LinkedList<Integer>();

				int sz = lists.size();
				minPos = checkIfMatched(lists, pointers);
				comparisons = comparisons + (sz - 1); // k-1 # of comparisons
														// for k linked list at
														// a time
				if (minPos.peekLast() == 1) // means matching
				{

					res.add(lists.get(i).get(pointers[i]));
					listExhausted = incrementAll(pointers, lists);
				} else {
					for (int j = 0; j < minPos.size() - 1; j++) // leaving the
																// last element
																// as its the
																// flag
					{
						int listNum = minPos.get(j);

						if (pointers[listNum] == lists.get(listNum).size() - 1) {
							/* Reached the last element jumping off */
							listExhausted = true;
							break;
						}

						pointers[listNum] += 1;
					}
				}

			}
		}

		FileUtils.write(f, "DaatAnd\n", StandardCharsets.UTF_8, true);
		FileUtils.write(f, qterm + "\n", StandardCharsets.UTF_8, true);
		FileUtils.write(f, "Results: ", StandardCharsets.UTF_8, true);
		String doc = "";
		if (res.size() > 0) {
			for (int j = 0; j < res.size()-1; j++) {
				doc = doc + res.get(j) + " ";
			}
			doc = doc + res.getLast();
		} else
			doc = doc + "empty";
		doc = doc + "\n";
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of documents in results: %d\n", res.size());
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of comparisons: %d\n", comparisons);
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
	}

	LinkedList<Integer> checkIfMatchedOr(LinkedList<LinkedList<Integer>> lists, int[] pointers, int[] siz) {
		LinkedList<Integer> minList = new LinkedList<Integer>();
		int minVal = 99999;
		int currId = 0;
		boolean flag = true;
		int val, l = 0;
		for (int k = 0; k < pointers.length; k++) {
			// System.out.println(k+" pointer "+pointers[k]+" siz "+siz[k]);
			if (pointers[k] < siz[k]) {
				l = k;
				break;
			}
		}

		val = lists.get(l).get(pointers[l]);
		// System.out.println("l "+l+" val "+val);
		for (int i = 1; i < lists.size(); i++) {
			if (pointers[i] >= siz[i])
				continue;
			currId = lists.get(i).get(pointers[i]);
			if (val != currId) {
				flag = false;
			}
		}

		for (int i = 0; i < lists.size(); i++) {
			if (pointers[i] >= siz[i])
				continue;
			currId = lists.get(i).get(pointers[i]);
			if (currId < minVal) {
				minVal = currId;
			}
		}

		for (int i = 0; i < lists.size(); i++) {
			if (pointers[i] >= siz[i])
				continue;
			currId = lists.get(i).get(pointers[i]);
			if (minVal == currId) {
				minList.add(i);
			}
		}
		// System.out.println("minVal "+ minVal+" minList "+minList);
		if (flag)
			minList.add(1);
		else
			minList.add(0);

		return minList;
	}

	public void daatOR(HashMap<String, LinkedList<Integer>> map, String qterm, File f) throws Exception {

		String[] q = qterm.split(" ");
		LinkedList<LinkedList<Integer>> lists = new LinkedList<LinkedList<Integer>>();
		for (int i = 0; i < q.length; i++) {
			lists.add(map.get(q[i]));
		}
		int[] pointers = new int[lists.size()];
		for (int i = 0; i < pointers.length; i++)
			pointers[i] = 0;
		int comparisons = 0;

		LinkedList<Integer> res = new LinkedList<Integer>();
		res.clear();
		boolean jumpOff = false; /* For jumping out of the indefinite loop */
		/*
		 * DAATOR logic is a little tricky. The code will loop indefinitely
		 * until k-1 linked list have exhausted At the end last non-exhausted
		 * list would be appended to optimize the number of comparisons
		 */

		do {
			/*
			 * Store the size of the linked list for checking if the list
			 * exhausted at runtime
			 */
			int siz[] = new int[lists.size()];
			for (int i = 0; i < lists.size(); i++) {
				siz[i] = lists.get(i).size();
				// System.out.println(lists.get(i));
				// System.out.print("pointer["+i+"] "+pointers[i]+"\n");
			}

			for (int i = 0; i < lists.size(); i++) {
				int exhCount = 0; /*
									 * To count the number of exhausted list so
									 * far
									 */

				for (int n = 0; n < pointers.length; n++) {
					if (pointers[n] >= siz[n]) {
						exhCount++;
					}
				}
				// System.out.println(exhCount+" number of lists exhausted ");
				if (exhCount >= lists.size() - 1) {
					// System.out.println(exhCount+" number of lists exhausted
					// so exiting!!");
					jumpOff = true;
					break;
				}
				// System.out.println(exhCount+" exhC "+i+"andi "+pointers[i]);
				if (pointers[i] >= siz[i]) {
					exhCount++;
					// System.out.println(i+" list exhasued so skipping loop");
					continue;
				}

				LinkedList<Integer> minList = new LinkedList<Integer>();
				/*
				 * Get the enumerations of the linked list with minimum value of
				 * docId It is a list becuase there is a possibility of multiple
				 * lists (say j lists) with same docId which turns out to be
				 * minimum than other k-j lists so following function returns
				 * the list
				 */
				minList = checkIfMatchedOr(lists, pointers, siz);
				// System.out.println("checkIfMatchedOr returned "+minList);

				comparisons += (lists.size() - 1 - exhCount);
				/*
				 * Like TAATOR logic the last value in the list would be the
				 * flag for determining if all the list have the same docId. In
				 * this case, all the list's pointers would be incremented at
				 * once to optimize the number of comparisons
				 */
				if (minList.peekLast() == 1) {
					int docId = lists.get(i).get(pointers[i]);
					// System.out.println("Adding "+docId+" to the result");
					res.add(docId);

					for (int k = 0; k < lists.size(); k++) {
						int iD = lists.get(k).get(pointers[k]);
						if ((iD == docId) && (pointers[k] <= siz[k])) {
							pointers[k] += 1;

						}
					}
				} else {
					int valAtMinPos = 0;
					for (int j = 0; j < minList.size() - 1; j++) {
						int listNum = minList.get(j);
						valAtMinPos = lists.get(listNum).get(pointers[listNum]);
						if ((pointers[listNum] <= siz[listNum])) {
							pointers[listNum] += 1;
						}
					}
					// System.out.println("Adding "+valAtMinPos+" to th
					// result");
					res.add(valAtMinPos);// Add only once

				} // else
			}
			if (jumpOff) {
				/*
				 * Append the last left list to optimize the number of
				 * comparisons
				 */
				for (int i = 0; i < pointers.length; i++) {
					if (pointers[i] >= siz[i])
						continue;
					while (pointers[i] < siz[i]) {
						int docID = lists.get(i).get(pointers[i]);
						// System.out.println("adding leftover "+docID+" from
						// "+i+" pointer["+pointers[i]+"]");
						res.add(docID);
						pointers[i] += 1;
					}
				}
				break;
			}
			/*
			 * Need to loop it indefinitely untill all the lists have been
			 * exhausted and processed
			 */
		} while (true);

		FileUtils.write(f, "DaatOr\n", StandardCharsets.UTF_8, true);
		FileUtils.write(f, qterm + "\n", StandardCharsets.UTF_8, true);
		FileUtils.write(f, "Results: ", StandardCharsets.UTF_8, true);
		String doc = "";
		for (int j = 0; j < res.size()-1; j++) {
			doc = doc + res.get(j) + " ";

		}
		doc = doc + res.getLast();
		doc = doc + "\n";
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of documents in results: %d\n", res.size());
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
		doc = String.format("Number of comparisons: %d\n", comparisons);
		FileUtils.write(f, doc, StandardCharsets.UTF_8, true);
	}

	public static void main(String[] args) throws Exception {
		String out_file = args[1];
		String inp_file = args[2];
		App obj = new App(args[0]);
		HashMap<String, LinkedList<Integer>> map = obj.parse();
		List<String> query = FileUtils.readLines(new File(inp_file), StandardCharsets.UTF_8);
		File file = new File(out_file);
		FileUtils.write(file, "", StandardCharsets.UTF_8, false);
		for (String s : query) {
			String[] input = s.split(" ");
			if (input.length < 2) {
				System.out.println("Atleast two terms required to perform AND, OR operations");
				FileUtils.write(file, "Atleast two terms required to perform AND, OR operations",
						StandardCharsets.UTF_8, false);
			}

			obj.taatAND(map, s, file);
			obj.taatOR(map, s, file);
			obj.daatAND(map, s, file);
			obj.daatOR(map, s, file);

		}
		map.clear();

	}
}