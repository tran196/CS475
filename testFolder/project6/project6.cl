kernel void ArrayMult( global const float *dA, global const float *dB, global float *dC )
{
        int gid = get_global_id( 0 );

        dC[gid] = dA[gid] * dB[gid];
}

kernel void ArrayMultAdd( global const float *dA, global const float *dB, global float *dC, global const float *dD )
{
        int gid = get_global_id( 0 );

        dC[gid] = dA[gid] * dB[gid] + dD[gid];
}


kernel void ArrayMultReduce( global const float *dA, global const float *dB, local float *products, global float *dC )
{
        int gid = get_global_id( 0 );
        int lid = get_local_id( 0 );
        int totalItems = get_local_size( 0 );

        int groupID = get_group_id( 0 );
        products[ lid ] = dA[ gid ] * dB[ gid ];

        for ( int offset = 1; offset < totalItems; offset *= 2 ) {
                int mask = 2 * offset - 1;
                barrier( CLK_LOCAL_MEM_FENCE );
                if( ( lid & mask ) == 0 ) {
                        products[ lid ] += products[ lid + offset ];
                }
        }

        barrier( CLK_LOCAL_MEM_FENCE );
        if( lid == 0 ) {
            dC[ groupID ] = products[ 0 ];
        }

}
