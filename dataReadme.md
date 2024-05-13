| Item | Input  |	Output | 功能 |
|:--------:|:--------:|:--------:|:--------:|
| extract_number | string | score in that string | fetch the score in that line |
| papers | -  | - | a list stores all the papers' review info | 
| papers.forum | papers[id]['forum'] | string of forum name | - |  
| papers.authorids | papers[id]['authorids'] | string of authorids| get authors of the id-th paper |
| papers.decision | papers[id]['decision'] | Accepted or Rejected | get whether the id-th paper is accepted |
| papers.scores | papers[id]['scores'] | scores of different reviews | get the list of the id-th paper's score |