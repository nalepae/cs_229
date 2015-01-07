%% display_data: Display X 2D data in a grid
%% X is a matrix where each row represents an image.
%% Each row of X represents an image
%% item_width is the width of each image.
%% If not specified, this function assumes that each image has a square shape.
function [h, display_array] = display_data(X, item_width)
	% Compute m : Number of items to dysplay
	% Compute n : Number of pixels for each item
	[m n] = size(X); 

	% Set a default value to example width if not specified in the
	% calling function.
	% In this case, each item will be considered as a square
	if ~exist('item_width', 'var')
		item_width = round(sqrt(n));
	end

	% Gray Image
	colormap(gray);

	% Compute item height
	item_height = n / item_width;

	% Compute the number of items to display
	% Examples : If 9 items to display, the display matrix will be [3 * 3]
	%            If 10 items to display, the display matrix will be [3 * 4]
	%            If 11 items to display, the display matrix will be [3 * 4]
	%            If 12 items to display, the display matrix will be [3 * 4]
	%            If 13 items to display, the display matrix will be [3 * 5]
	%            If 14 items to display, the display matrix will be [3 * 5]
	%            If 15 items to display, the display matrix will be [3 * 5]
	%            If 16 items to display, the display matrix will be [4 * 4]
	% ...
	% ...
	% ...
	display_rows = floor(sqrt(m));
	display_cols = ceil(m / display_rows);
	
	% Padding between each image
	pad = 1;

	% Setup a blank display
	nb_row_pixels = pad + display_rows * (item_height + pad);
	nb_col_pixels = pad + display_cols * (item_width + pad);

	display_array =  - ones(nb_row_pixels, nb_col_pixels);

	% Compute the matrix to display
	for j = 1:m
		% Compute the row number and the column number of item
		item_row = floor((j - 1) / display_cols) + 1;
		item_col = mod(j - 1, display_cols) + 1;

		first_pixel_row = pad + (item_row - 1) * (item_height + pad) + 1;
		first_pixel_col = pad + (item_col - 1) * (item_width + pad) + 1;

		% Compute the maximum value of the item
		max_value = max(X(j, :));

		% Transform the item from vector shape to matrix shape
		matrix_item = reshape(X(j, :), item_height, item_width) / max_value;

		% Put this matrix item inside the big matrix 'display_array'
		row_vec = first_pixel_row:(first_pixel_row + item_height - 1);
		col_vec = first_pixel_col:(first_pixel_col + item_width - 1);

		display_array(row_vec, col_vec) = matrix_item;
	end

	% Display image (-1 : black, 1 : white)
	imagesc(display_array, [-1 ,1]);
end