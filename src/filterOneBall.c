

#define MAX 20 

int solid_ball[MAX] = {0}; 

int getZeroCount(int n){
    int k = 0;
    for (int i = 0; i < n;++i){
        if (solid_ball[i] == 0)
            k++;
    }
    return k;
}

int filterOneBall(int n){
    for (int i = 0; i < MAX;++i){
        solid_ball[i] = 0;
    }
    int i = 1;
    int j = 1; 
    int k = 0;
    while (getZeroCount(n) != 1){
        i = 1;
        j = 1;
        for (k = 0;k < n;++k){
            if (solid_ball[k] == 1)
                i -= 1;
            if (i == j * j && solid_ball[k] != 1){
                j += 1;
                solid_ball[k] = 1;
            }
            i += 1;  
        }                 
    }
    for (int i = 0; i < n;++i){
        if (solid_ball[i] == 0)
            return i;
    }
    return -1;
}
