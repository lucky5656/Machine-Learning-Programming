function word_indices = processEmail(email_contents)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices 
%   word_indices = PROCESSEMAIL(email_contents) preprocesses 
%   the body of an email and returns a list of indices of the 
%   words contained in the email. 
%

% Load Vocabulary
vocabList = getVocabList();%getVocabList()读取vocab.txt中的固定词汇表，并返回单词的单元数组（单元矩阵）

% Init return value
word_indices = [];

% ========================== Preprocess Email ===========================
% Find the Headers ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);

% Lower case
email_contents = lower(email_contents);%lower(str)将str中的所有大写字符转换为相应的小写字符,并保留所有其他字符不变。

% Strip all HTML
% Looks for any expression that starts with < and ends with > and replace
% and does not have any < or > in the tag it with a space
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');%regexprep(str,expression,replace)将str中与expression匹配的文本替换为replace描述的文本

% Handle Numbers
% Look for one or more characters between 0-9
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% Handle URLS
% Look for strings starting with http:// or https://
email_contents = regexprep(email_contents, '(http|https)://[^\s]*', 'httpaddr');

% Handle Email Addresses
% Look for strings with @ in the middle
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Handle $ sign
email_contents = regexprep(email_contents, '[$]+', 'dollar');

% ========================== Tokenize Email ===========================
% Output the email to screen as well
fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;

while ~isempty(email_contents)

    % Tokenize and also get rid of any punctuation(标记并去掉任何标点符号)
    [str, email_contents] = strtok(email_contents, [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);% strtok()把字符串以规定的字符分割开
                          %[token,remain] = strtok(str,delimiters)使用delimiters中的分隔字符解析str,返回标文和剩余文本,char(13) char(10)是换了两行
   
    % Remove any non alphanumeric characters(删除任何非字母数字字符)
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try 
        str = porterStemmer(strtrim(str)); %porterStemmer()波特词干算法,调用此函数,使用要词干化的字符串作为输入参数,此函数将词干单词作为字符串返回。
    catch
        str = ''; %捕获异常
        continue;
    end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end

    % Look up the word in the dictionary and add to word_indices if found（在字典中查找该单词，如果找到则添加到单词索引中）
    % ====================== YOUR CODE HERE ======================
    % Instructions: Fill in this function to add the index of str to
    %               word_indices if it is in the vocabulary. At this point
    %               of the code, you have a stemmed word from the email in
    %               the variable str. You should look up str in the
    %               vocabulary list (vocabList). If a match exists, you
    %               should add the index of the word to the word_indices
    %               vector. Concretely, if str = 'action', then you should
    %               look up the vocabulary list to find where in vocabList
    %               'action' appears. For example, if vocabList{18} =
    %               'action', then, you should add 18 to the word_indices 
    %               vector (e.g., word_indices = [word_indices ; 18]; ).
    % 
    % Note: vocabList{idx} returns a the word with index idx in the
    %       vocabulary list.
    % 
    % Note: You can use strcmp(str1, str2) to compare two strings (str1 and
    %       str2). It will return 1 only if the two strings are equivalent.
    %

    for i=1:length(vocabList)
        if( strcmp(vocabList{i}, str) )%strcmp(s1,s1)是用于做字符串比较的函数,如果s1和s1是一致的，则返回1，否则返回0
          word_indices = [word_indices;i];%将所有索引的值存入word_indices
        end
    end
    % =============================================================

    % Print to screen, ensuring that the output lines are not too long
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;

end

% Print footer
fprintf('\n\n=========================\n');

end
