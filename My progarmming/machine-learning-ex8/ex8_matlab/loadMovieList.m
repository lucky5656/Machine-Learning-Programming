function movieList = loadMovieList()
%GETMOVIELIST reads the fixed movie list in movie.txt and returns a
%cell array of the words
%   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
%   and returns a cell array of the words in movieList.


%% Read the fixed movieulary list
fid = fopen('movie_ids.txt');

% Store all movies in cell array movie{}
n = 1682;  % Total number of movies 

movieList = cell(n, 1);
for i = 1:n
    % Read line
    line = fgets(fid);%fgets函数用于读取文件中指定一行，并保留换行符，与之前的fopen配套使用
    % Word Index (can ignore since it will be = i)
    [idx, movieName] = strtok(line, ' ');%按' '分类捕捉索引和字符串电影名变量
    % Actual Word
    movieList{i} = strtrim(movieName);%此函数返回字符串str的副本并删除了所有前导和尾随空格字符
end
fclose(fid);

end
