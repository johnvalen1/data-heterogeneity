PGDMP          ;                |            data_heterog %   14.10 (Ubuntu 14.10-0ubuntu0.22.04.1) %   14.10 (Ubuntu 14.10-0ubuntu0.22.04.1)     +           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            ,           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            -           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            .           1262    17226    data_heterog    DATABASE     a   CREATE DATABASE data_heterog WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE = 'en_CA.UTF-8';
    DROP DATABASE data_heterog;
                postgres    false            �            1259    17231    datasets    TABLE     k   CREATE TABLE public.datasets (
    id bigint NOT NULL,
    dataset_name character varying(255) NOT NULL
);
    DROP TABLE public.datasets;
       public         heap    postgres    false            �            1259    17230    datasets_id_seq    SEQUENCE     x   CREATE SEQUENCE public.datasets_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 &   DROP SEQUENCE public.datasets_id_seq;
       public          postgres    false    210            /           0    0    datasets_id_seq    SEQUENCE OWNED BY     C   ALTER SEQUENCE public.datasets_id_seq OWNED BY public.datasets.id;
          public          postgres    false    209            �            1259    17269    results    TABLE     �  CREATE TABLE public.results (
    id bigint NOT NULL,
    instance_id integer NOT NULL,
    dataset_id integer NOT NULL,
    create_date date NOT NULL,
    n_c0 integer,
    n_c1 integer,
    n_c2 integer,
    n_c3 integer,
    n_c0_control integer,
    n_c1_control integer,
    n_c2_control integer,
    n_c3_control integer,
    n_c0_treatment integer,
    n_c1_treatment integer,
    n_c2_treatment integer,
    n_c3_treatment integer,
    intra_d_c0 real,
    intra_d_c1 real,
    intra_d_c2 real,
    intra_d_c3 real,
    acc_c0 real,
    auc_c0 real,
    f1_c0 real,
    prec_c0 real,
    recall_c0 real,
    acc_c1 real,
    auc_c1 real,
    f1_c1 real,
    prec_c1 real,
    recall_c1 real,
    acc_c2 real,
    auc_c2 real,
    f1_c2 real,
    prec_c2 real,
    recall_c2 real,
    acc_c3 real,
    auc_c3 real,
    f1_c3 real,
    prec_c3 real,
    recall_c3 real,
    dist_c0_c1 real,
    dist_c1_c2 real,
    dist_c2_c3 real,
    dist_c0_c2 real,
    dist_c0_c3 real,
    dist_c1_c3 real,
    c0_cross_fold_time real,
    c1_cross_fold_time real,
    c2_cross_fold_time real,
    c3_cross_fold_time real,
    sample_size_id integer NOT NULL
);
    DROP TABLE public.results;
       public         heap    postgres    false            �            1259    17268    results_id_seq    SEQUENCE     w   CREATE SEQUENCE public.results_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 %   DROP SEQUENCE public.results_id_seq;
       public          postgres    false    212            0           0    0    results_id_seq    SEQUENCE OWNED BY     A   ALTER SEQUENCE public.results_id_seq OWNED BY public.results.id;
          public          postgres    false    211            �            1259    17283    sample_sizes    TABLE     �   CREATE TABLE public.sample_sizes (
    id bigint NOT NULL,
    dataset_id integer NOT NULL,
    size_n integer,
    n_control integer,
    n_treatment integer
);
     DROP TABLE public.sample_sizes;
       public         heap    postgres    false            �            1259    17282    sample_sizes_id_seq    SEQUENCE     |   CREATE SEQUENCE public.sample_sizes_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 *   DROP SEQUENCE public.sample_sizes_id_seq;
       public          postgres    false    214            1           0    0    sample_sizes_id_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.sample_sizes_id_seq OWNED BY public.sample_sizes.id;
          public          postgres    false    213            �           2604    17234    datasets id    DEFAULT     j   ALTER TABLE ONLY public.datasets ALTER COLUMN id SET DEFAULT nextval('public.datasets_id_seq'::regclass);
 :   ALTER TABLE public.datasets ALTER COLUMN id DROP DEFAULT;
       public          postgres    false    209    210    210            �           2604    17272 
   results id    DEFAULT     h   ALTER TABLE ONLY public.results ALTER COLUMN id SET DEFAULT nextval('public.results_id_seq'::regclass);
 9   ALTER TABLE public.results ALTER COLUMN id DROP DEFAULT;
       public          postgres    false    211    212    212            �           2604    17286    sample_sizes id    DEFAULT     r   ALTER TABLE ONLY public.sample_sizes ALTER COLUMN id SET DEFAULT nextval('public.sample_sizes_id_seq'::regclass);
 >   ALTER TABLE public.sample_sizes ALTER COLUMN id DROP DEFAULT;
       public          postgres    false    213    214    214            $          0    17231    datasets 
   TABLE DATA           4   COPY public.datasets (id, dataset_name) FROM stdin;
    public          postgres    false    210   �       &          0    17269    results 
   TABLE DATA           _  COPY public.results (id, instance_id, dataset_id, create_date, n_c0, n_c1, n_c2, n_c3, n_c0_control, n_c1_control, n_c2_control, n_c3_control, n_c0_treatment, n_c1_treatment, n_c2_treatment, n_c3_treatment, intra_d_c0, intra_d_c1, intra_d_c2, intra_d_c3, acc_c0, auc_c0, f1_c0, prec_c0, recall_c0, acc_c1, auc_c1, f1_c1, prec_c1, recall_c1, acc_c2, auc_c2, f1_c2, prec_c2, recall_c2, acc_c3, auc_c3, f1_c3, prec_c3, recall_c3, dist_c0_c1, dist_c1_c2, dist_c2_c3, dist_c0_c2, dist_c0_c3, dist_c1_c3, c0_cross_fold_time, c1_cross_fold_time, c2_cross_fold_time, c3_cross_fold_time, sample_size_id) FROM stdin;
    public          postgres    false    212   �       (          0    17283    sample_sizes 
   TABLE DATA           V   COPY public.sample_sizes (id, dataset_id, size_n, n_control, n_treatment) FROM stdin;
    public          postgres    false    214   �2       2           0    0    datasets_id_seq    SEQUENCE SET     =   SELECT pg_catalog.setval('public.datasets_id_seq', 1, true);
          public          postgres    false    209            3           0    0    results_id_seq    SEQUENCE SET     =   SELECT pg_catalog.setval('public.results_id_seq', 29, true);
          public          postgres    false    211            4           0    0    sample_sizes_id_seq    SEQUENCE SET     A   SELECT pg_catalog.setval('public.sample_sizes_id_seq', 4, true);
          public          postgres    false    213            �           2606    17236    datasets datasets_pkey 
   CONSTRAINT     T   ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_pkey PRIMARY KEY (id);
 @   ALTER TABLE ONLY public.datasets DROP CONSTRAINT datasets_pkey;
       public            postgres    false    210            �           2606    17274    results results_pkey 
   CONSTRAINT     R   ALTER TABLE ONLY public.results
    ADD CONSTRAINT results_pkey PRIMARY KEY (id);
 >   ALTER TABLE ONLY public.results DROP CONSTRAINT results_pkey;
       public            postgres    false    212            �           2606    17288    sample_sizes sample_sizes_pkey 
   CONSTRAINT     \   ALTER TABLE ONLY public.sample_sizes
    ADD CONSTRAINT sample_sizes_pkey PRIMARY KEY (id);
 H   ALTER TABLE ONLY public.sample_sizes DROP CONSTRAINT sample_sizes_pkey;
       public            postgres    false    214            $      x�3����SpN�KN-����� 6��      &      x�m�[v�HD�齴N�|mbV��_�� ([��k��"(2�G )��?�<�i����_{��m\k^;�����k�˜�q�5/;�eq��l�����U�u��~�����^s�6l��Y�E��Za]G˶������=���y���Y_�Gc�����n\�vV����6�ny���ɣ1�Z|��ǫg��������3�ZG��zm��K�x���/ۯ�z�x��O�~����}����8վ����1��2��>�[�m��c�Z<?;�m�
�m�x��.����:���Cm)"��rS�v����k�L��@b#���ޟ����!'�y�WD���Ĺ�����]k֗��鉤�<��L�x\0�j��<Ѽ�郕��G����lmO�)�6w�>���v9]9d-�����5��+6���W?�s��W4r|\���߯~�_�0V�"��_��kL�a.w�v�T��1�2
��e�����d<�Vm#�SD��[�J?y�c߇��XO����|WB_�B�GO��ݧ��>�б��hu1*��ȏ}����Nd�U�T:s��>-�<˕[�a��{�z�ZK�L��I�*>$���<��K�|�����j���_q�o�"���,HVrS���]&Q��,yU�ƝC���"k�Z�ȣ�j�!��������6�,z�k>�Z�'}�7�lz�7�Q���ɧ}ی�__������)��@Y���J�f�����p�S7�ݲ��͉WT ��[L�K�� ��\�y�;�K�
x&��x$P�ɯ��@���hص��L!�����qբ�����ō&��qK)Ut�g4H��, ��Y.�H�2��>�~&��m��oX�w0���q���(	�k��	�,�]8h���Ʊ�G����>V��Jy^��}e�W���{u��"�<��t�w��f����`,oč�tE�T�P��ٕ�kЍf[�0pe�%Or2�}�U3S��38�&��#6Qrݖ㮎BP�:Ia�˳
���j�� ����e�-O iԱ�h�y6
��������N�OD�=cgz��C}�{��) ��7Z�.Ns���a�lI�Y�'$�Z$�y�zYm~�+V��Zh�	��:��+�đw�0ZZz@�h>d_�NF����h���}R�iߔ[v|�I`� ũe8�@�+�i�pu� j�yӷ�Xg) =I�J��GX��Eߝ�<�d�O�_�E�m7?˪���y� U+�m��g���$J;+��&ѫ`������)�Ft9��c�S���Y�0�Y�\$V��A�V�a�l-������z{��(nܽ���'��>��a�+)�7�s�r�6~d�S�=��q��-ت��2���n�!�hL������"��f*�
�\���p*�ej�N�羞̦�es�ud'vѺ^���(���dYP��_�� ��@�mV	Ln(�y�|J��3��3U�Q��&�S_9D	���C%��F�8�� }��Q�E�dϊ�r	�E�<*q�4r�[�-(��]VE)b
B��RCL�� ��m�(��t�2�5��"p�'�E?t$B5�e��v��E��V\l��$@�B�!�/b��~�y�[Ho��
d�p���č
�J@o�U5���b0p?u�>C(�!��"��6!�۟^�I:Hy��3�K��i	"�ܬ`���W�ڭ[���ἰ71�v� �����<G'%ْ����NQ�1��J9�t\����Dj[�O����Z$�A��j`��L�gu�+�x�j�\X��~��3c���ÊE݇Yن�M=���LMr�CS3ճ{��0�7#ޢX,��'�qRz�r���gX LA9�b�yLs��0��Ƀ�"C o�*ȯbAڸj�ŏ�Vʻ��PI%���l�v�j����U������Û��������RW�6Zk;�荭!'{aݷ9��:x˲���-q��{ ��X�2j�G:��ZV/�jq���܊��Ǘ ݷ`M�� �������F'I`=
���W�����9�"$3���C�7�S{V�Cα$��w"�D��g�`<�EH��M�@�R^d���1a�iV�ߩ����]yOѥ��0��-��R)�ú��#4v{�ɗ��b�t�Q��[Vu�F��a>�U��"�t+�wK���.�E��4\âT<F��y�y�/,��C�m�N$:���IT��=�r�����L���p	Q۫]*�Pdv�����J&�0��"e�f���声�VCQ�ͼ�8YƟ�ʨ��|�"\S�o+;�B�-YC�RJ�U?@���˾��Ȅ�u�[Aϲ3Q��3�o��=�Bw�����	�E��]>@Ƃ>Ds�{h�#-�T�@�X4qR$4r ��~ ���
�M D�jXB�]����.��ڒTI�
���Ty�{r)(O�?����ˁ�O��v2MI?5�� ��no�RӨuz�J�\��H�ѭu7hӎ'ǽx�(����ҟ�91+�SEb0��V�E�į�oC�#�ũM�:�A	��k�bI��8�84��jF��)�Dg���La����Ô��e��B��T5z�`J/��$�"�ݬ� :[u��m5M�&��V�����i��LQ��n��]2B�K8�|�ɕVcZ�v��8��7Q5ٺ����R3�/�Pwa���L�h5d��#�H��#7jf��z���W(�C���xWe�;�-�v���RXw��J����FO� }(z�N�p�����д�f@�F*\5Q-GÆ� 5-��J��y�,�.(vBV�M?�F�.��cƣ��$�E�y�J��W}�Q�Qk$�4�SU���6��j�o3j �l44�u�D�#�W��!�7$��Ģ��C�ͺ��qH	���L/Z�M{<c�+�xY����4��`�T�eB�V)�J%�k��N�\�$B��)�$��=c�#���E
�Ih���N�:��T�|�U��T`�(�;Q�u�@�����,�bӜ/^��7�6!1���p�1�7����/�WK�yz�ڨд��DKu��>A��O8C�K%sf��If�
�*���+Iup���K�psD��G�6S�Ϯ��b�zj�����ࡎ��@,��A�}�W�	4ov�}�0���5��1��e���W�	zwџ�	�Y����\�Kڽɥ��Ì3x3V��N0���s�]��[)jē�rZcdm?�h��,MtwV�<���@@CB�XT������D�D�x��a�}a�k�ʻ� 6Գ�.�-���(�bo�AM$�t�-���.��j8�Z v��u�(���=�o��t�~0����o��I��g�;�A#����5ѣ$5k��d����f�F%�"kI�����������4Ԑ>�w��u^NӼ'䭆���k�t��šM��1���`�������TCf����@�5���E\r7V��'����|r�������{���JW�
Zۺw!n�Ł��ry�wAiJ���4G�e�}!�D�,K�T��@zL��%x��jI܆��w�̜% �}&0�6�&E��4LPT�3�/$"M��d�������W�.���/����l
P�>�h�ڪ�O�x���@xE�.�Z���Cm�[�[�=?O"�O3�jE6H޴[��n���-�{E�f\����,Ҥ��b�O����v1K����/52�X>��q �)gQ���sH�k��t�tӾ垹��VN��@E��n�Dۊ�F�0�=������wk`�,D'���#����>�xb�foP��dk�oJ�r��	5�*�{c�0jh�j�yoC/���AH��B�����L�R{}C��Dx��n8JON�E�B0S�15c�'�lQoU,=7y4Mmb�&���g +7lI3}2v�S_�ҙO��˯�9�M,�Z�99e�������j��]ύ�x��{�p�6O|OMV��������F]	����~��#o��_#%�����w|	�S\U��AC��4��+Lb��u��d��g4��#�К΄w�j��KMµ3�Yq��sL�5<T�ė c��\��&I�i��bQ\\��m�D�U�E�䥒ʩyL� M���Ŭ�2�!R �  �js��	��O4i� i�T;[�C�^<͋S�P4��������O{��7ߪ�~ؕ��H����b����m�k�p4�6�_�|����l9j�jP��]h3WL������&P��RS���r����z%aZ9ΆZ^Ibs���uw���������G���[��jitв�נDs��q��oՆ�y;���(z�ۨ}�!"�kJ�@��=�\�B<����^��6K<j�y䏏`�d�~I��6$g��}�p�E#9K˂ĭ(�C�P����Xr���`ӿV�6{�,��{7��aj$r�9�f���9@ek�Ƅ�|��n�EE���������T-J�jv[��m�;���2*�Zr�]( �%a.XB�*���� �|�棡��4k��L����<����H���yh��z�hH#n�C>O��K"�|/N�i��YCA���H��(�VJJ��Wr��!6MFE�r��JSpm�����|�SE��n�Dq9�6�4�����]#�����eN�6lc�w׭^I��Yg�O^�v����ۤ1�KΙ� ;Ȥ�A�ƣ���-P]�l�.08����A���imó��?�`���8�n��Jb�ʍ���__�6��֤G=y�~$�15��%�~�=���Ȳ4hz��ƍ�Jn�j̑u���ɿ��I����QV�0����:f��U�X�5�&ͧ'�/�`�S�F�08�4�M
�#�S�~��C	�z�K�)G�Z����������F���      (   ?   x�ȱ 1�X*�ǒ��^�Op�� lwA�E=�-���r-��e�;�3�G���     